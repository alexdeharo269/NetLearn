// MC_NullModel.cpp  –  Memory Capacity vs null surrogate networks
//
// Null models (all preserve topology, degree, strength):
//   1. Broken-stick  (Barrat et al. 2004) – random Beta(1,k-1) weights
//   2. Row-uniform                         – all edges get s_i/k_i
//   3. Global reshuffle                    – permute all non-zero weights
//
// Compile (M4 Mac, homebrew libomp):
//   g++ -std=c++20 -O3 -Xpreprocessor -fopenmp \
//       -I/opt/homebrew/opt/libomp/include \
//       -I/opt/homebrew/include/eigen3 \
//       -L/opt/homebrew/opt/libomp/lib -lomp \
//       MC_NullModel.cpp ESN.o Math.o Utils.o -o mc_null

#include <cstdlib>
#include <tuple>
#include <iostream>
#include <fstream>
#include <format>
#include <memory>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "ESN.h"
#include "Task.h"

using namespace std;

// ── Feature flags ─────────────────────────────────────────────────────────────
constexpr bool ENABLE_LEGENDRE  = true;
constexpr bool ENABLE_UNIFORM   = true;   // row-uniform null
constexpr bool ENABLE_RESHUFFLE = true;   // global-reshuffle null
constexpr int  LEG_TAU1         = 10;
constexpr int  LEG_TAU2         = 5;
constexpr int  NULL_REPS        = 10;     // surrogate realisations per null model
constexpr int  WIN_REPS         = 5;      // Win realisations to average MC over

// ── Forward declarations ──────────────────────────────────────────────────────
double CalculateMemoryCapacity(const vector<vector<double>> &X_states,
                               const vector<double> &predictions,
                               int tau_vals, int size_input, double train_ratio);
double CalculateNARMAError(const vector<vector<double>> &X_states,
                           const vector<double> &predictions,
                           int size_input, double train_ratio);
tuple<vector<vector<vector<double>>>, vector<int>> process_csv(ifstream &, int);

// ── Surrogate generators ──────────────────────────────────────────────────────

// 1. Broken-stick: Beta(1,k-1) weights, preserves s_i per row
vector<vector<double>> surrogateBrokenStick(const vector<vector<double>>& W, mt19937& rng)
{
    int n = (int)W.size();
    vector<vector<double>> Wbs(n, vector<double>(n, 0.0));
    uniform_real_distribution<double> uni(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        vector<int> edges;
        double strength = 0.0;
        for (int j = 0; j < n; ++j)
            if (W[i][j] != 0.0) { edges.push_back(j); strength += W[i][j]; }

        int k = (int)edges.size();
        if (k == 0) continue;
        if (k == 1) { Wbs[i][edges[0]] = strength; continue; }

        vector<double> bp(k - 1);
        for (auto& b : bp) b = uni(rng) * strength;
        sort(bp.begin(), bp.end());

        Wbs[i][edges[0]] = bp[0];
        for (int m = 1; m < k - 1; ++m)
            Wbs[i][edges[m]] = bp[m] - bp[m - 1];
        Wbs[i][edges[k - 1]] = strength - bp[k - 2];
    }
    return Wbs;
}

// 2. Row-uniform: every edge gets exactly s_i / k_i
//    Strongest null — no weight heterogeneity whatsoever within each node
vector<vector<double>> surrogateRowUniform(const vector<vector<double>>& W)
{
    int n = (int)W.size();
    vector<vector<double>> Wu(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        vector<int> edges;
        double strength = 0.0;
        for (int j = 0; j < n; ++j)
            if (W[i][j] != 0.0) { edges.push_back(j); strength += W[i][j]; }

        if (edges.empty()) continue;
        double w_uniform = strength / (double)edges.size();
        for (int j : edges) Wu[i][j] = w_uniform;
    }
    return Wu;
}

// 3. Global reshuffle: collect all non-zero weights, shuffle, reassign
//    Preserves exact weight distribution but breaks spatial assignment
vector<vector<double>> surrogateReshuffle(const vector<vector<double>>& W, mt19937& rng)
{
    int n = (int)W.size();
    vector<vector<double>> Wr(n, vector<double>(n, 0.0));

    // Collect all (i,j) edge positions and their weights
    vector<pair<int,int>> positions;
    vector<double> weights;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (W[i][j] != 0.0) {
                positions.emplace_back(i, j);
                weights.push_back(W[i][j]);
            }

    // Shuffle weights, keep positions (topology unchanged)
    shuffle(weights.begin(), weights.end(), rng);
    for (int e = 0; e < (int)positions.size(); ++e)
        Wr[positions[e].first][positions[e].second] = weights[e];

    return Wr;
}

// ── State collection ──────────────────────────────────────────────────────────
vector<vector<double>> collectStates(ESN& esn, const vector<double>& sequence,
                                     int size_input, int res_size)
{
    vector<vector<double>> X(size_input, vector<double>(res_size, 0.0));
    esn.state.assign(res_size, 0.0);
    for (int t = 0; t < size_input; ++t) {
        esn.input[0] = sequence[t];
        esn.EvoluteReservoir();
        X[t] = esn.state;
    }
    return X;
}

// ── Average MC over multiple Win realisations ─────────────────────────────────
// Takes a matrix W, builds WIN_REPS ESNs with that W (different Win each time),
// returns mean MC. ESNs must be built with the shared rng → call SERIALLY.
// We pre-build and store them; this helper just averages states + MC.
double avgMC(const vector<unique_ptr<ESN>>& esns,
             const vector<double>& sequence,
             const vector<double>& predictions,
             int tau_vals, int size_input, double train_ratio, int res_size)
{
    double acc = 0.0;
    for (const auto& esn : esns) {
        // collectStates needs non-const ESN (modifies state in place)
        auto& e = const_cast<ESN&>(*esn);
        auto X  = collectStates(e, sequence, size_input, res_size);
        acc    += CalculateMemoryCapacity(X, predictions, tau_vals, size_input, train_ratio);
    }
    return acc / (double)esns.size();
}

// Build WIN_REPS ESNs with the given W. Must be called from serial context.
vector<unique_ptr<ESN>> buildWinReps(ESNParams params,
                                     const vector<vector<double>>& W,
                                     double sr)
{
    vector<unique_ptr<ESN>> v(WIN_REPS);
    for (auto& e : v) {
        e = make_unique<ESN>(params);
        e->setW(W);
        e->rescaleReservoir(sr);
    }
    return v;
}

// ── Result struct ─────────────────────────────────────────────────────────────
struct NullResult {
    int age;
    // Broken-stick
    double mc_g_real, mc_g_bs,   ratio_g_bs;
    double mc_b_real, mc_b_bs,   ratio_b_bs;
    // Row-uniform
    double mc_g_uni,  ratio_g_uni;
    double mc_b_uni,  ratio_b_uni;
    // Reshuffle
    double mc_g_rsh,  ratio_g_rsh;
    double mc_b_rsh,  ratio_b_rsh;
};

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    // ── ESN params ────────────────────────────────────────────────────────────
    ESNParams params_glob;
    params_glob.reservoir_size         = 90;
    params_glob.spectral_radius        = 0.99;
    params_glob.input_size             = 1;
    params_glob.tau_vals               = 15;
    params_glob.output_size            = 1;
    params_glob.connectivity_reservoir = 10;
    params_glob.target_nodes           = {};

    ESNParams params_bio = params_glob;
    params_bio.target_nodes = {{76, 77}};

    // ── Task ──────────────────────────────────────────────────────────────────
    TaskParams tp;
    tp.size_input  = 3000;
    tp.tau_vals    = 15;
    tp.train_ratio = 0.7;
    tp.type        = MEMORY_CAPACITY;
    Task t_mc(tp);

    // ── Load connectomes ──────────────────────────────────────────────────────
    ifstream infile("data/connectomes_sample.csv");
    
    vector<vector<vector<double>>> connectomes;
    vector<int> ages;
    tie(connectomes, ages) = process_csv(infile, params_glob.reservoir_size);
    int n = (int)ages.size();
    cout << format("Loaded {} connectomes\n", n);

    // ── Pre-build real ESNs SERIALLY — WIN_REPS per subject ──────────────────
    // Outer index: subject; inner: Win replicate
    vector<vector<unique_ptr<ESN>>> esns_g(n), esns_b(n);
    for (int i = 0; i < n; ++i) {
        esns_g[i] = buildWinReps(params_glob, connectomes[i], 0.99);
        esns_b[i] = buildWinReps(params_bio,  connectomes[i], 0.99);
    }

    // ── Row-uniform surrogates are deterministic → pre-build serially too ────
    vector<vector<unique_ptr<ESN>>> esns_g_uni(n), esns_b_uni(n);
    if constexpr (ENABLE_UNIFORM) {
        for (int i = 0; i < n; ++i) {
            auto Wu = surrogateRowUniform(connectomes[i]);
            esns_g_uni[i] = buildWinReps(params_glob, Wu, 0.99);
            esns_b_uni[i] = buildWinReps(params_bio,  Wu, 0.99);
        }
    }

    // ── Result buffer ─────────────────────────────────────────────────────────
    vector<NullResult> results(n);

    // ── Parallel subject loop ─────────────────────────────────────────────────
    atomic<int> done{0};
    #pragma omp parallel for schedule(dynamic, 1) num_threads(5)
    for (int i = 0; i < n; ++i)
    {
        mt19937 rng(42 + i * 97);   // per-thread, no shared state

        // ── Real MC (averaged over WIN_REPS) ──────────────────────────────────
        double mc_g = avgMC(esns_g[i], t_mc.sequence, t_mc.predictions,
                            tp.tau_vals, tp.size_input, tp.train_ratio,
                            params_glob.reservoir_size);
        double mc_b = avgMC(esns_b[i], t_mc.sequence, t_mc.predictions,
                            tp.tau_vals, tp.size_input, tp.train_ratio,
                            params_bio.reservoir_size);

        // ── Broken-stick null ─────────────────────────────────────────────────
        double mc_g_bs = 0.0, mc_b_bs = 0.0;
        for (int r = 0; r < NULL_REPS; ++r) {
            auto Wbs = surrogateBrokenStick(connectomes[i], rng);

            // Single Win per null rep (enough given NULL_REPS averaging)
            ESN eg_r(params_glob); eg_r.setW(Wbs); eg_r.rescaleReservoir(0.99);
            ESN eb_r(params_bio);  eb_r.setW(Wbs); eb_r.rescaleReservoir(0.99);

            auto Xgr = collectStates(eg_r, t_mc.sequence, tp.size_input, params_glob.reservoir_size);
            auto Xbr = collectStates(eb_r, t_mc.sequence, tp.size_input, params_bio.reservoir_size);

            mc_g_bs += CalculateMemoryCapacity(Xgr, t_mc.predictions, tp.tau_vals, tp.size_input, tp.train_ratio);
            mc_b_bs += CalculateMemoryCapacity(Xbr, t_mc.predictions, tp.tau_vals, tp.size_input, tp.train_ratio);
        }
        mc_g_bs /= NULL_REPS;
        mc_b_bs /= NULL_REPS;

        // ── Row-uniform null (pre-built, just read) ───────────────────────────
        double mc_g_uni = 0.0, mc_b_uni = 0.0;
        if constexpr (ENABLE_UNIFORM) {
            mc_g_uni = avgMC(esns_g_uni[i], t_mc.sequence, t_mc.predictions,
                             tp.tau_vals, tp.size_input, tp.train_ratio,
                             params_glob.reservoir_size);
            mc_b_uni = avgMC(esns_b_uni[i], t_mc.sequence, t_mc.predictions,
                             tp.tau_vals, tp.size_input, tp.train_ratio,
                             params_bio.reservoir_size);
        }

        // ── Global-reshuffle null ─────────────────────────────────────────────
        double mc_g_rsh = 0.0, mc_b_rsh = 0.0;
        if constexpr (ENABLE_RESHUFFLE) {
            for (int r = 0; r < NULL_REPS; ++r) {
                auto Wr = surrogateReshuffle(connectomes[i], rng);

                ESN eg_r(params_glob); eg_r.setW(Wr); eg_r.rescaleReservoir(0.99);
                ESN eb_r(params_bio);  eb_r.setW(Wr); eb_r.rescaleReservoir(0.99);

                auto Xgr = collectStates(eg_r, t_mc.sequence, tp.size_input, params_glob.reservoir_size);
                auto Xbr = collectStates(eb_r, t_mc.sequence, tp.size_input, params_bio.reservoir_size);

                mc_g_rsh += CalculateMemoryCapacity(Xgr, t_mc.predictions, tp.tau_vals, tp.size_input, tp.train_ratio);
                mc_b_rsh += CalculateMemoryCapacity(Xbr, t_mc.predictions, tp.tau_vals, tp.size_input, tp.train_ratio);
            }
            mc_g_rsh /= NULL_REPS;
            mc_b_rsh /= NULL_REPS;
        }

        auto safe_ratio = [](double a, double b){ return b > 0.0 ? a / b : 0.0; };

        results[i] = {
            ages[i],
            mc_g, mc_g_bs,  safe_ratio(mc_g, mc_g_bs),
            mc_b, mc_b_bs,  safe_ratio(mc_b, mc_b_bs),
            mc_g_uni, safe_ratio(mc_g, mc_g_uni),
            mc_b_uni, safe_ratio(mc_b, mc_b_uni),
            mc_g_rsh, safe_ratio(mc_g, mc_g_rsh),
            mc_b_rsh, safe_ratio(mc_b, mc_b_rsh),
        };

        #pragma omp critical
        cout << format("\rProcessed: {}/{}", ++done, n) << flush;
    }
    cout << "\n";

    // ── Write CSV ─────────────────────────────────────────────────────────────
    ofstream out("mc_null_results.csv");
    out << "subject_id,"
           "MC_Glob_Real,MC_Glob_BS,Ratio_Glob_BS,"
           "MC_Bio_Real,MC_Bio_BS,Ratio_Bio_BS";
    if constexpr (ENABLE_UNIFORM)
        out << ",MC_Glob_Uni,Ratio_Glob_Uni,MC_Bio_Uni,Ratio_Bio_Uni";
    if constexpr (ENABLE_RESHUFFLE)
        out << ",MC_Glob_Rsh,Ratio_Glob_Rsh,MC_Bio_Rsh,Ratio_Bio_Rsh";
    out << "\n";

    for (int i = 0; i < n; ++i) {
        const auto& r = results[i];
        out << format("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}",
                      r.age,
                      r.mc_g_real, r.mc_g_bs,  r.ratio_g_bs,
                      r.mc_b_real, r.mc_b_bs,  r.ratio_b_bs);
        if constexpr (ENABLE_UNIFORM)
            out << format(",{:.4f},{:.4f},{:.4f},{:.4f}",
                          r.mc_g_uni, r.ratio_g_uni,
                          r.mc_b_uni, r.ratio_b_uni);
        if constexpr (ENABLE_RESHUFFLE)
            out << format(",{:.4f},{:.4f},{:.4f},{:.4f}",
                          r.mc_g_rsh, r.ratio_g_rsh,
                          r.mc_b_rsh, r.ratio_b_rsh);
        out << "\n";
    }

    cout << "Done. Results in mc_null_results.csv\n";
    return 0;
}