// ipc_main.cpp  –  Information Processing Capacity (Dambre et al. 2012)
// Parallelized with OpenMP for Apple M4 (10-core).
//
// IPC terms computed:
//   Linear          Σ_τ      R²[ P1(u(t-τ))              ]
//   Quad            Σ_τ      R²[ P2(u(t-τ))              ]
//   Cross_11        Σ_{τ1<τ2} R²[ P1(u(t-τ1))·P1(u(t-τ2)) ]
//   Cross_21 (opt.) Σ_{τ1≠τ2} R²[ P2(u(t-τ1))·P1(u(t-τ2)) ]
//   Cross_22 (opt.) Σ_{τ1<τ2} R²[ P2(u(t-τ1))·P2(u(t-τ2)) ]
//
// Toggle higher-order cross terms:
//   ENABLE_CROSS_21 true/false
//   ENABLE_CROSS_22 true/false
//
// Compile (M4 Mac, homebrew libomp):
//   brew install libomp          # one-time
//   g++ -std=c++20 -O3 -Xpreprocessor -fopenmp \
//       -I/opt/homebrew/opt/libomp/include \
//       -I/opt/homebrew/include/eigen3 \
//       -L/opt/homebrew/opt/libomp/lib -lomp \
//       ipc_main.o ESN.o Math.o Utils.o -o ipc_test

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <format>
#include <functional>
#include <memory>
#include <atomic>
#include <omp.h>
#include <Eigen/Dense>
#include "ESN.h"
#include "Task.h"

using namespace std;

// ── Feature flags ─────────────────────────────────────────────────────────────
// Cross_21: P2(u(t-τ1))·P1(u(t-τ2)),  τ1 ≠ τ2  → τ_max² fits
// Cross_22: P2(u(t-τ1))·P2(u(t-τ2)),  τ1 < τ2  → τ_max*(τ_max-1)/2 fits
// With τ_max=20: Cross_11=190, Cross_21=400, Cross_22=190 → 780 fits total
constexpr bool ENABLE_CROSS_21 = false;
constexpr bool ENABLE_CROSS_22 = false;

// ── Forward declarations ───────────────────────────────────────────────────────
double computeSpectralRadius_Internal(const vector<vector<double>> &matrix);
Eigen::VectorXd TrainReadout(const Eigen::MatrixXd &, const Eigen::VectorXd &, double);
double PearsonCorrelation(const vector<double> &, const vector<double> &);
tuple<vector<vector<vector<double>>>, vector<int>> process_csv(ifstream &, int);

// ── Legendre basis (L2-orthogonal on U(-1,1)) ─────────────────────────────────
inline double P1(double u) { return u; }
inline double P2(double u) { return (3.0 * u * u - 1.0) / 2.0; }

// ── State collection ───────────────────────────────────────────────────────────
vector<vector<double>> collectStates(ESN &esn, const Task &task,
                                     const TaskParams &tp, int res_size)
{
    vector<vector<double>> states(tp.size_input, vector<double>(res_size, 0.0));
    esn.state.assign(res_size, 0.0);
    for (int t = 0; t < tp.size_input; ++t) {
        esn.input[0] = task.sequence[t];
        esn.EvoluteReservoir();
        states[t] = esn.state;
    }
    return states;
}

// ── Train/test split ──────────────────────────────────────────────────────────
struct Split {
    Eigen::MatrixXd X_train, X_test;
    int washout, train_size, test_size, train_offset;
};

Split buildSplit(const vector<vector<double>> &states, int size_input, double train_ratio)
{
    const int washout = 100;
    int res   = states[0].size();
    int n_tr  = (int)(size_input * train_ratio) - washout;
    int n_te  = size_input - (int)(size_input * train_ratio);
    int t_off = washout + n_tr;

    Split s{ {}, {}, washout, n_tr, n_te, t_off };
    s.X_train.resize(n_tr, res + 1);
    s.X_test.resize (n_te, res + 1);

    for (int i = 0; i < n_tr; ++i) {
        for (int j = 0; j < res; ++j) s.X_train(i, j) = states[washout + i][j];
        s.X_train(i, res) = 1.0;
    }
    for (int i = 0; i < n_te; ++i) {
        for (int j = 0; j < res; ++j) s.X_test(i, j) = states[t_off + i][j];
        s.X_test(i, res) = 1.0;
    }
    return s;
}

// ── Single Ridge fit → R² ─────────────────────────────────────────────────────
double scoreTarget(const Split &s, const function<double(int)> &target_fn)
{
    Eigen::VectorXd Y_tr(s.train_size);
    vector<double>  Y_te(s.test_size);

    for (int i = 0; i < s.train_size; ++i) Y_tr(i) = target_fn(s.washout + i);
    for (int i = 0; i < s.test_size;  ++i) Y_te[i]  = target_fn(s.train_offset + i);

    Eigen::VectorXd pred = s.X_test * TrainReadout(s.X_train, Y_tr, 5e-3);
    vector<double> pred_vec(s.test_size);
    for (int i = 0; i < s.test_size; ++i) pred_vec[i] = pred(i);

    double r = PearsonCorrelation(Y_te, pred_vec);
    return r;
}

// ── IPC result ────────────────────────────────────────────────────────────────
struct IPCResult {
    double linear   = 0.0;   // Σ R²[P1(u(t-τ))]
    double quad     = 0.0;   // Σ R²[P2(u(t-τ))]
    double cross_11 = 0.0;   // Σ_{τ1<τ2}  R²[P1·P1]
    double cross_21 = 0.0;   // Σ_{τ1≠τ2}  R²[P2·P1]  (optional)
    double cross_22 = 0.0;   // Σ_{τ1<τ2}  R²[P2·P2]  (optional)
};

// ── IPC computation ───────────────────────────────────────────────────────────
IPCResult IPCtest(const TaskParams &tp, const ESNParams &params,
                  ESN &esn, const Task &task)
{
    auto   states  = collectStates(esn, task, tp, params.reservoir_size);
    Split  s       = buildSplit(states, tp.size_input, tp.train_ratio);
    const  auto &u = task.sequence;
    int    tau_max = (int)tp.tau_vals;

    auto U = [&](int t) -> double { return (t >= 0) ? u[t] : 0.0; };

    // OpenMP reduction requires plain scalars — collect into locals, assign at end.
    double lin = 0.0, quad = 0.0, cross_11 = 0.0, cross_21 = 0.0, cross_22 = 0.0;

    // ── Diagonal terms: tau_max fits each, fully independent ─────────────────
    for (int tau = 1; tau <= tau_max; ++tau) {
        lin  += scoreTarget(s, [&, tau](int t) { return P1(U(t - tau)); });
        quad += scoreTarget(s, [&, tau](int t) { return P2(U(t - tau)); });
    }

    // ── Cross_11: P1×P1, upper triangle, τ1 < τ2  (190 fits @ τ_max=20) ─────
    #pragma omp parallel for collapse(2) reduction(+:cross_11) \
        schedule(dynamic, 4) if(!omp_in_parallel())
    for (int tau1 = 1; tau1 <= tau_max; ++tau1)
        for (int tau2 = 1; tau2 <= tau_max; ++tau2) {
            if (tau2 <= tau1) continue;
            cross_11 += scoreTarget(s, [&, tau1, tau2](int t) {
                return P1(U(t - tau1)) * P1(U(t - tau2));
            });
        }

    // ── Cross_21: P2×P1, all ordered pairs τ1≠τ2  (400 fits @ τ_max=20) ─────
    if constexpr (ENABLE_CROSS_21) {
        #pragma omp parallel for collapse(2) reduction(+:cross_21) \
            schedule(dynamic, 4) if(!omp_in_parallel())
        for (int tau1 = 1; tau1 <= tau_max; ++tau1)
            for (int tau2 = 1; tau2 <= tau_max; ++tau2) {
                if (tau1 == tau2) continue;
                cross_21 += scoreTarget(s, [&, tau1, tau2](int t) {
                    return P2(U(t - tau1)) * P1(U(t - tau2));
                });
            }
    }

    // ── Cross_22: P2×P2, upper triangle, τ1 < τ2  (190 fits @ τ_max=20) ─────
    if constexpr (ENABLE_CROSS_22) {
        #pragma omp parallel for collapse(2) reduction(+:cross_22) \
            schedule(dynamic, 4) if(!omp_in_parallel())
        for (int tau1 = 1; tau1 <= tau_max; ++tau1)
            for (int tau2 = 1; tau2 <= tau_max; ++tau2) {
                if (tau2 <= tau1) continue;
                cross_22 += scoreTarget(s, [&, tau1, tau2](int t) {
                    return P2(U(t - tau1)) * P2(U(t - tau2));
                });
            }
    }

    return { lin, quad, cross_11, cross_21, cross_22 };
}

// ── CSV header helper (compile-time, avoids repetition) ───────────────────────
string csvHeader(const string &suffix)
{
    string h = format("Lin_{0},Quad_{0},Cross11_{0},", suffix);
    if constexpr (ENABLE_CROSS_21) h += format("Cross21_{0},", suffix);
    if constexpr (ENABLE_CROSS_22) h += format("Cross22_{0},", suffix);
    return h;
}

string csvRow(const IPCResult &r)
{
    string s = format("{:.4f},{:.4f},{:.4f},", r.linear, r.quad, r.cross_11);
    if constexpr (ENABLE_CROSS_21) s += format("{:.4f},", r.cross_21);
    if constexpr (ENABLE_CROSS_22) s += format("{:.4f},", r.cross_22);
    return s;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    // Print active configuration
    cout << format("IPC terms: Lin, Quad, Cross_11"
                   "{}{}\n",
                   ENABLE_CROSS_21 ? ", Cross_21" : "",
                   ENABLE_CROSS_22 ? ", Cross_22" : "");

    // ── ESN params ────────────────────────────────────────────────────────────
    ESNParams params_glob;
    params_glob.reservoir_size         = 90;
    params_glob.spectral_radius        = 0.99;
    params_glob.input_size             = 1;
    params_glob.tau_vals               = 20;   // Cross_11+22: 190 fits each
                                               // Cross_21:    400 fits
    params_glob.output_size            = 1;
    params_glob.connectivity_reservoir = 10;
    params_glob.target_nodes           = {};

    ESNParams params_bio = params_glob;
    params_bio.target_nodes = {{76, 77}};

    // ── Task: MEMORY_CAPACITY generates U(-1,1) sequence ─────────────────────
    TaskParams tp;
    tp.size_input  = 10000;
    tp.tau_vals    = params_glob.tau_vals;
    tp.train_ratio = 0.7;
    tp.type        = MEMORY_CAPACITY;
    Task task(tp);

    // ── Load connectomes ──────────────────────────────────────────────────────
    ifstream infile("data/connectomes_sample.csv");
    auto [connectomes, ages] = process_csv(infile, params_glob.reservoir_size);
    int n = (int)ages.size();
    cout << format("Loaded {} connectomes\n", n);

    // ── Pre-build all ESNs SERIALLY (ESN::rng is a shared static) ────────────
    vector<unique_ptr<ESN>> esns_g(n), esns_b(n);
    for (int i = 0; i < n; ++i) {
        esns_g[i] = make_unique<ESN>(params_glob);
        esns_g[i]->setW(connectomes[i]);
        esns_g[i]->rescaleReservoir(0.99);

        esns_b[i] = make_unique<ESN>(params_bio);
        esns_b[i]->setW(connectomes[i]);
        esns_b[i]->rescaleReservoir(0.99);
    }
    connectomes.clear();

    // ── Result buffers ────────────────────────────────────────────────────────
    vector<IPCResult> res_g(n), res_b(n);

    // ── L1 parallelism: subject loop ─────────────────────────────────────────
    atomic<int> done{0};
    #pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads())
    for (int i = 0; i < n; ++i) {
        res_g[i] = IPCtest(tp, params_glob, *esns_g[i], task);
        res_b[i] = IPCtest(tp, params_bio,  *esns_b[i], task);

        #pragma omp critical
        cout << format("\rProcessed: {}/{}", ++done, n) << flush;
    }
    cout << "\n";

    // ── Write CSV ─────────────────────────────────────────────────────────────
    ofstream outfile("ipc_results.csv");
    outfile << "subject_id," << csvHeader("Glob") << csvHeader("Bio") << "\n";
    for (int i = 0; i < n; ++i)
        outfile << format("{},", ages[i])
                << csvRow(res_g[i])
                << csvRow(res_b[i]) << "\n";

    cout << "Done. Results in ipc_results.csv\n";
    return 0;
}