// esn_mc — plain Memory Capacity for every subject (the main experiment).
// in : data/connectomes.csv  [subject_id, flattened NxN]
// out: MC_Glob per subject. Outer loop over subjects is OpenMP-parallel.
#include "esn.hpp"
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.txt";
    Config cfg(cfg_path);

    ESNParams p;
    p.N           = cfg.geti("reservoir_size", 90);
    p.steps       = cfg.geti("steps", 6000);
    p.washout     = cfg.geti("washout", 100);
    p.n_win       = cfg.geti("win_reps", 1);
    p.rho         = cfg.getd("spectral_radius", 0.99);
    p.ridge       = cfg.getd("ridge", 1e-4);
    p.train_ratio = cfg.getd("train_ratio", 0.7);
    p.tau_max     = cfg.geti("tau", 20);
    p.seed        = (unsigned)cfg.geti("seed", 42);

    std::string in_csv  = cfg.gets("in_csv",  "data/connectomes.csv");
    std::string out_csv = cfg.gets("out_csv", "mc_results.csv");
    int threads = cfg.geti("threads", 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::vector<int> ids; std::vector<MatrixXd> mats;
    read_connectomes(in_csv, p.N, ids, mats);
    const int S = (int)mats.size();
    std::cerr << "esn_mc: " << S << " subjects | N=" << p.N << " steps=" << p.steps
              << " tau=" << p.tau_max << " win_reps=" << p.n_win << "\n";

    std::vector<double> mc(S, 0.0);
    auto t0 = std::chrono::steady_clock::now();
    long done = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; ++i) {
        std::mt19937 rng(p.seed + 1000u * (unsigned)ids[i]);
        mc[i] = mc_total(mats[i], p, rng);
        #pragma omp atomic
        ++done;
        if (done % 200 == 0) {
            #pragma omp critical
            std::cerr << "  " << done << "/" << S << "\n";
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "esn_mc done in "
              << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s\n";

    std::ofstream out(out_csv);
    out << "subject_id,MC_Glob\n";
    for (int i = 0; i < S; ++i) out << ids[i] << "," << mc[i] << "\n";
    std::cerr << "wrote " << out_csv << "\n";
    return 0;
}
