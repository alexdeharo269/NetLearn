// esn_ipc — Information Processing Capacity order decomposition (Legendre P1,
// P2, optional P1*P1 cross) at small tau, on a small subject subset.
#include "esn.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.txt";
    Config cfg(cfg_path);

    ESNParams p;
    p.N=cfg.geti("reservoir_size",90); p.steps=cfg.geti("steps",6000);
    p.washout=cfg.geti("washout",100); p.n_win=1;
    p.rho=cfg.getd("spectral_radius",0.99); p.ridge=cfg.getd("ridge",1e-4);
    p.train_ratio=cfg.getd("train_ratio",0.7); p.tau_max=cfg.geti("tau",20);
    p.seed=(unsigned)cfg.geti("seed",42);
    int  ipc_tau  = cfg.geti("ipc_tau", 6);
    bool do_cross = cfg.getb("ipc_cross", false);

    std::string in_csv  = cfg.gets("in_csv",  "data/connectomes_sub.csv");
    std::string out_csv = cfg.gets("out_csv", "ipc_results.csv");
    int threads = cfg.geti("threads", 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::vector<int> ids; std::vector<MatrixXd> mats;
    read_connectomes(in_csv, p.N, ids, mats);
    const int S = (int)mats.size();
    std::cerr << "esn_ipc: " << S << " subjects | ipc_tau=" << ipc_tau
              << " cross=" << do_cross << "\n";

    std::vector<std::vector<std::string>> names(S);
    std::vector<VectorXd> vals(S);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; ++i) {
        std::mt19937 rng(p.seed + 1000u * (unsigned)ids[i]);
        std::vector<std::string> nm;
        vals[i] = ipc_terms(mats[i], p, ipc_tau, do_cross, rng, nm);
        names[i] = nm;
        #pragma omp critical
        std::cerr << "  subject " << (i + 1) << "/" << S << "\r";
    }
    std::cerr << "\n";

    std::ofstream out(out_csv);
    out << "subject_id";
    for (auto& n : names[0]) out << "," << n;
    out << "\n";
    for (int i = 0; i < S; ++i) {
        out << ids[i];
        for (int j = 0; j < vals[i].size(); ++j) out << "," << vals[i](j);
        out << "\n";
    }
    std::cerr << "wrote " << out_csv << "\n";
    return 0;
}
