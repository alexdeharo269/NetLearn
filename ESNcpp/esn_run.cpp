// ─────────────────────────────────────────────────────────────────────────────
// esn_run.cpp  —  Memory-Capacity engine for the thesis (mode = "mc")
//
// Computes, PER SUBJECT and in ONE pass over the connectome list:
//   * MC_Glob               real MC, input projected to ALL nodes        (PRIMARY)
//   * MC_Bio                real MC, input projected to thalamic nodes    (PRIMARY)
//   * MC_Glob_r  (optional) Σ|r| variant of MC_Glob                       (check: r vs r²)
//   * three weight surrogates × {global, bio} + their MC ratios:
//        BS  broken-stick   (random split of node strength)   = disparity null model
//        Uni row-uniform    (every edge = s_i / k_i)          = "H0" no weight heterogeneity
//        Rsh reshuffle      (node-level by default)           = "H1" wiring-vs-weights
//   * thalamic random-pair null: K random input pairs → MC_Rand_*, mean/std, Z_Score
//
// Everything that the thesis tunes is read from config.txt (see Config.h); NOTHING is
// hard-coded, so tau / ridge / train-test / steps / replicates are edited from the
// notebook's first cell. The PRIMARY metric is MC = Σ_τ r²(τ) with tau = 20.
//
// Parallelism: the OUTER loop over subjects is OpenMP-parallel. ESNs are built INSIDE
// the parallel region with a per-instance seed (ESNParams::seed), so there is no shared
// mutable RNG and peak memory stays ~O(threads) ESNs instead of O(subjects).
//
// Windows / MSYS2-mingw64 build (one line; eigen3 via `pacman -S mingw-w64-x86_64-eigen3`):
//   g++ -std=c++20 -O3 -fopenmp -I/mingw64/include/eigen3 \
//       ESN.cpp Math.cpp Utils.cpp esn_run.cpp -o esn_run.exe
// ─────────────────────────────────────────────────────────────────────────────
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
#include <vector>
#include <omp.h>
#include "ESN.h"
#include "Task.h"
#include "Config.h"

using namespace std;

// ── From the shared library (Math.cpp / Utils.cpp) ───────────────────────────
double CalculateMemoryCapacity(const vector<vector<double>> &X_states,
                               const vector<double> &predictions,
                               int tau_vals, int size_input, double train_ratio,
                               double ridge, bool squared, int washout);
tuple<vector<int>, vector<vector<vector<double>>>>
readConnectomesWithID(std::ifstream &infile, int size, bool zero_diagonal);

// ── State collection (drive reservoir with the MC input sequence) ────────────
static vector<vector<double>> collectStates(ESN &esn, const vector<double> &sequence,
                                            int size_input, int res_size)
{
    vector<vector<double>> X(size_input, vector<double>(res_size, 0.0));
    esn.state.assign(res_size, 0.0);
    for (int t = 0; t < size_input; ++t)
    {
        esn.input[0] = sequence[t];
        esn.EvoluteReservoir();
        X[t] = esn.state;
    }
    return X;
}

// ── Mean MC over WIN_REPS random input projections of the SAME reservoir W ───
// Each replicate builds a fresh ESN with a deterministic per-instance seed
// (thread-safe). Returns the average MC (r² by default).
static double mcForW(ESNParams base, const vector<vector<double>> &W,
                     const Task &task, const TaskParams &tp,
                     double ridge, int washout, bool squared,
                     int win_reps, unsigned seed_base)
{
    double acc = 0.0;
    for (int r = 0; r < win_reps; ++r)
    {
        ESNParams p = base;
        p.seed = seed_base + (unsigned)r;          // distinct, reproducible Win each replicate
        ESN esn(p);
        esn.setW(W);
        esn.rescaleReservoir(base.spectral_radius); // normalise ρ(W) before driving
        auto X = collectStates(esn, task.sequence, tp.size_input, base.reservoir_size);
        acc += CalculateMemoryCapacity(X, task.predictions, tp.tau_vals,
                                       tp.size_input, tp.train_ratio, ridge, squared, washout);
    }
    return acc / (double)win_reps;
}

// ── Surrogate generators (preserve topology + node strength/degree) ──────────
// 1. Broken-stick: split each node strength by k-1 uniform breakpoints → the
//    disparity null model (random normalized weights over neighbours).
static vector<vector<double>> surrBrokenStick(const vector<vector<double>> &W, mt19937 &rng)
{
    int n = (int)W.size();
    vector<vector<double>> Wbs(n, vector<double>(n, 0.0));
    uniform_real_distribution<double> uni(0.0, 1.0);
    for (int i = 0; i < n; ++i)
    {
        vector<int> e; double s = 0.0;
        for (int j = 0; j < n; ++j) if (W[i][j] != 0.0) { e.push_back(j); s += W[i][j]; }
        int k = (int)e.size();
        if (k == 0) continue;
        if (k == 1) { Wbs[i][e[0]] = s; continue; }
        vector<double> bp(k - 1);
        for (auto &b : bp) b = uni(rng) * s;
        sort(bp.begin(), bp.end());
        Wbs[i][e[0]] = bp[0];
        for (int m = 1; m < k - 1; ++m) Wbs[i][e[m]] = bp[m] - bp[m - 1];
        Wbs[i][e[k - 1]] = s - bp[k - 2];
    }
    return Wbs;
}
// 2. Row-uniform: every existing edge of node i becomes s_i / k_i (no heterogeneity).
static vector<vector<double>> surrRowUniform(const vector<vector<double>> &W)
{
    int n = (int)W.size();
    vector<vector<double>> Wu(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        vector<int> e; double s = 0.0;
        for (int j = 0; j < n; ++j) if (W[i][j] != 0.0) { e.push_back(j); s += W[i][j]; }
        if (e.empty()) continue;
        double w = s / (double)e.size();
        for (int j : e) Wu[i][j] = w;
    }
    return Wu;
}
// 3a. Node-level reshuffle: permute the weights WITHIN each node's edges (paper "H1").
static vector<vector<double>> surrReshuffleNode(const vector<vector<double>> &W, mt19937 &rng)
{
    int n = (int)W.size();
    vector<vector<double>> Wr(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
    {
        vector<int> e; vector<double> w;
        for (int j = 0; j < n; ++j) if (W[i][j] != 0.0) { e.push_back(j); w.push_back(W[i][j]); }
        shuffle(w.begin(), w.end(), rng);
        for (size_t m = 0; m < e.size(); ++m) Wr[i][e[m]] = w[m];
    }
    return Wr;
}
// 3b. Global reshuffle: permute all non-zero weights across the whole network.
static vector<vector<double>> surrReshuffleGlobal(const vector<vector<double>> &W, mt19937 &rng)
{
    int n = (int)W.size();
    vector<vector<double>> Wr(n, vector<double>(n, 0.0));
    vector<pair<int,int>> pos; vector<double> w;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (W[i][j] != 0.0) { pos.emplace_back(i, j); w.push_back(W[i][j]); }
    shuffle(w.begin(), w.end(), rng);
    for (size_t e = 0; e < pos.size(); ++e) Wr[pos[e].first][pos[e].second] = w[e];
    return Wr;
}

int main(int argc, char **argv)
{
    Config cfg;
    cfg.load(argc > 1 ? argv[1] : "config.txt");

    // ── Read every knob from config (defaults match the thesis PRIMARY setting) ──
    const int    N        = cfg.geti("reservoir_size", 90);
    const double SR        = cfg.getd("spectral_radius", 0.99);
    const int    TAU       = cfg.geti("tau", 20);                 // PRIMARY = 20
    const double RIDGE     = cfg.getd("ridge", 1e-4);
    const double TRAIN     = cfg.getd("train_ratio", 0.7);
    const int    STEPS     = cfg.geti("steps", 6000);            // input sequence length
    const int    WASHOUT   = cfg.geti("washout", 100);
    const int    WIN_REPS  = cfg.geti("win_reps", 5);
    const int    NULL_REPS = cfg.geti("null_reps", 10);
    const int    N_RAND    = cfg.geti("n_rand_pairs", 0);        // thalamic random-pair null (0 = off)
    const unsigned SEED    = (unsigned)cfg.geti("seed", 42);
    const bool   DO_R      = cfg.getb("do_r", false);            // also compute Σ|r| variant
    const bool   DO_BS     = cfg.getb("do_bs", true);
    const bool   DO_UNI    = cfg.getb("do_uni", true);
    const bool   DO_RSH    = cfg.getb("do_rsh", true);
    const string RSH_MODE  = cfg.gets("reshuffle_mode", "node"); // "node" (H1) | "global"
    const string IN_CSV    = cfg.gets("in_csv",  "data/connectomes.csv");
    const string OUT_CSV   = cfg.gets("out_csv", "mc_results.csv");
    vector<int>  THAL      = cfg.getints("thal_nodes");          // e.g. 76,77
    const int    NTHREADS  = cfg.geti("threads", omp_get_max_threads());

    // ── Parameter sets: global vs bio (thalamic) input projection ────────────
    ESNParams pG;
    pG.reservoir_size = N; pG.spectral_radius = SR; pG.input_size = 1; pG.output_size = 1;
    pG.connectivity_reservoir = 10; pG.tau_vals = TAU; pG.target_nodes = {};

    ESNParams pB = pG;
    if (!THAL.empty()) pB.target_nodes = { THAL };   // single input dim → all thalamic nodes

    TaskParams tp; tp.type = MEMORY_CAPACITY; tp.size_input = STEPS; tp.tau_vals = TAU; tp.train_ratio = TRAIN;
    Task task(tp);   // shared, read-only during the parallel loop

    ifstream infile(IN_CSV);
    if (!infile) { cerr << "ERROR: cannot open " << IN_CSV << "\n"; return 1; }
    auto [ids, conns] = readConnectomesWithID(infile, N, /*zero_diagonal=*/true);
    int  n = (int)ids.size();
    cout << format("Loaded {} connectomes from {}\n", n, IN_CSV);
    cout << format("MC mode | tau={} ridge={} train={} steps={} win_reps={} null_reps={} threads={}\n",
                   TAU, RIDGE, TRAIN, STEPS, WIN_REPS, NULL_REPS, NTHREADS);

    // ── Per-subject result row ───────────────────────────────────────────────
    struct Row {
        int id;
        double mc_g = 0, mc_b = 0, mc_g_r = 0;
        double mc_g_bs = 0, mc_b_bs = 0, mc_g_uni = 0, mc_b_uni = 0, mc_g_rsh = 0, mc_b_rsh = 0;
        double mc_thal = 0, mc_rand_mean = 0, mc_rand_std = 0, z = 0;
        vector<double> mc_rand;
    };
    vector<Row> rows(n);

    auto reshuffle = [&](const vector<vector<double>> &W, mt19937 &rng) {
        return RSH_MODE == "global" ? surrReshuffleGlobal(W, rng) : surrReshuffleNode(W, rng);
    };

    atomic<int> done{0};
    #pragma omp parallel for schedule(dynamic, 1) num_threads(NTHREADS)
    for (int i = 0; i < n; ++i)
    {
        const auto &W = conns[i];
        Row r; r.id = ids[i];
        unsigned base = SEED + (unsigned)ids[i] * 100u;        // reproducible per subject
        mt19937 rng(base);                                      // per-thread surrogate RNG

        // ── Real MC (global / bio) ────────────────────────────────────────────
        r.mc_g = mcForW(pG, W, task, tp, RIDGE, WASHOUT, true,  WIN_REPS, base + 1u);
        r.mc_b = mcForW(pB, W, task, tp, RIDGE, WASHOUT, true,  WIN_REPS, base + 1u);
        if (DO_R) r.mc_g_r = mcForW(pG, W, task, tp, RIDGE, WASHOUT, false, WIN_REPS, base + 1u);

        // ── Broken-stick (disparity null) ─────────────────────────────────────
        if (DO_BS) {
            double g = 0, b = 0;
            for (int rep = 0; rep < NULL_REPS; ++rep) {
                auto Ws = surrBrokenStick(W, rng);
                g += mcForW(pG, Ws, task, tp, RIDGE, WASHOUT, true, 1, base + 10u + rep);
                b += mcForW(pB, Ws, task, tp, RIDGE, WASHOUT, true, 1, base + 10u + rep);
            }
            r.mc_g_bs = g / NULL_REPS; r.mc_b_bs = b / NULL_REPS;
        }
        // ── Row-uniform (deterministic surrogate; average over Win only) ──────
        if (DO_UNI) {
            auto Wu = surrRowUniform(W);
            r.mc_g_uni = mcForW(pG, Wu, task, tp, RIDGE, WASHOUT, true, WIN_REPS, base + 2000u);
            r.mc_b_uni = mcForW(pB, Wu, task, tp, RIDGE, WASHOUT, true, WIN_REPS, base + 2000u);
        }
        // ── Reshuffle (node-level by default) ─────────────────────────────────
        if (DO_RSH) {
            double g = 0, b = 0;
            for (int rep = 0; rep < NULL_REPS; ++rep) {
                auto Ws = reshuffle(W, rng);
                g += mcForW(pG, Ws, task, tp, RIDGE, WASHOUT, true, 1, base + 3000u + rep);
                b += mcForW(pB, Ws, task, tp, RIDGE, WASHOUT, true, 1, base + 3000u + rep);
            }
            r.mc_g_rsh = g / NULL_REPS; r.mc_b_rsh = b / NULL_REPS;
        }
        // ── Thalamic random-pair null ─────────────────────────────────────────
        if (N_RAND > 0) {
            r.mc_thal = r.mc_b;                       // MC_Bio is the thalamic input MC
            r.mc_rand.resize(N_RAND, 0.0);
            uniform_int_distribution<int> pick(0, N - 1);
            double sum = 0, sum2 = 0;
            for (int p = 0; p < N_RAND; ++p) {
                int a = pick(rng), c = pick(rng);
                while (c == a) c = pick(rng);
                ESNParams pr = pG; pr.target_nodes = { { a, c } };
                double m = mcForW(pr, W, task, tp, RIDGE, WASHOUT, true, WIN_REPS, base + 4000u + p);
                r.mc_rand[p] = m; sum += m; sum2 += m * m;
            }
            r.mc_rand_mean = sum / N_RAND;
            double var = sum2 / N_RAND - r.mc_rand_mean * r.mc_rand_mean;
            r.mc_rand_std = var > 0 ? sqrt(var) : 0.0;
            r.z = r.mc_rand_std > 0 ? (r.mc_thal - r.mc_rand_mean) / r.mc_rand_std : 0.0;
        }

        rows[i] = std::move(r);
        #pragma omp critical
        cout << format("\rProcessed: {}/{}", ++done, n) << flush;
    }
    cout << "\n";

    // ── Write CSV (subject_id first; columns gated by the same flags) ────────
    ofstream out(OUT_CSV);
    auto safe = [](double a, double b) { return b > 0.0 ? a / b : 0.0; };

    out << "subject_id,MC_Glob,MC_Bio";
    if (DO_R)  out << ",MC_Glob_r";
    if (DO_BS) out << ",MC_Glob_BS,MC_Bio_BS,Ratio_Glob_BS,Ratio_Bio_BS";
    if (DO_UNI)out << ",MC_Glob_Uni,MC_Bio_Uni,Ratio_Glob_Uni,Ratio_Bio_Uni";
    if (DO_RSH)out << ",MC_Glob_Rsh,MC_Bio_Rsh,Ratio_Glob_Rsh,Ratio_Bio_Rsh";
    if (N_RAND > 0) {
        out << ",MC_Thal,MC_Rand_Mean,MC_Rand_Std,Z_Score";
        for (int p = 0; p < N_RAND; ++p) out << format(",MC_Rand_{}", p);
    }
    out << "\n";

    for (const auto &r : rows) {
        out << format("{},{:.6f},{:.6f}", r.id, r.mc_g, r.mc_b);
        if (DO_R)  out << format(",{:.6f}", r.mc_g_r);
        if (DO_BS) out << format(",{:.6f},{:.6f},{:.6f},{:.6f}", r.mc_g_bs, r.mc_b_bs,
                                  safe(r.mc_g, r.mc_g_bs), safe(r.mc_b, r.mc_b_bs));
        if (DO_UNI)out << format(",{:.6f},{:.6f},{:.6f},{:.6f}", r.mc_g_uni, r.mc_b_uni,
                                  safe(r.mc_g, r.mc_g_uni), safe(r.mc_b, r.mc_b_uni));
        if (DO_RSH)out << format(",{:.6f},{:.6f},{:.6f},{:.6f}", r.mc_g_rsh, r.mc_b_rsh,
                                  safe(r.mc_g, r.mc_g_rsh), safe(r.mc_b, r.mc_b_rsh));
        if (N_RAND > 0) {
            out << format(",{:.6f},{:.6f},{:.6f},{:.6f}", r.mc_thal, r.mc_rand_mean, r.mc_rand_std, r.z);
            for (double m : r.mc_rand) out << format(",{:.6f}", m);
        }
        out << "\n";
    }
    cout << format("Done. Results in {}\n", OUT_CSV);
    return 0;
}
