// esn_surrogates — MC of the real network vs three weight null models, over
// several realizations, on a small subject subset. Produces tidy long-format
// output for mean+std boxplots. Goal: show the real/null MC gap is ~constant.
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
    p.washout=cfg.geti("washout",100); p.n_win=1;          // realization loop does the averaging
    p.rho=cfg.getd("spectral_radius",0.99); p.ridge=cfg.getd("ridge",1e-4);
    p.train_ratio=cfg.getd("train_ratio",0.7); p.tau_max=cfg.geti("tau",20);
    p.seed=(unsigned)cfg.geti("seed",42);
    int n_real = cfg.geti("n_real", 10);

    std::string in_csv  = cfg.gets("in_csv",  "data/connectomes_sub.csv");
    std::string out_csv = cfg.gets("out_csv", "surrogate_results.csv");
    int threads = cfg.geti("threads", 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::vector<int> ids; std::vector<MatrixXd> mats;
    read_connectomes(in_csv, p.N, ids, mats);
    const int S = (int)mats.size();
    std::cerr << "esn_surrogates: " << S << " subjects x " << n_real << " realizations\n";

    // one result row per (subject, realization, model)
    struct Row { int sid, rep; const char* model; double mc; };
    std::vector<std::vector<Row>> buf(S);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; ++i) {
        std::mt19937 rng(p.seed + 1000u * (unsigned)ids[i]);
        for (int rep = 0; rep < n_real; ++rep) {
            // Real: re-seed per realization so Real and nulls see the same input draw stream.
            buf[i].push_back({ids[i], rep, "Real",        mc_total(mats[i],                       p, rng)});
            buf[i].push_back({ids[i], rep, "BrokenStick", mc_total(null_brokenstick(mats[i], rng), p, rng)});
            buf[i].push_back({ids[i], rep, "Uniform",     mc_total(null_uniform(mats[i]),          p, rng)});
            buf[i].push_back({ids[i], rep, "Reshuffle",   mc_total(null_reshuffle(mats[i], rng),   p, rng)});
            buf[i].push_back({ids[i], rep, "SignFlip",    mc_total(null_signflip(mats[i], rng),    p, rng)});
        }
        #pragma omp critical
        std::cerr << "  subject " << (i + 1) << "/" << S << "\r";
    }
    std::cerr << "\n";

    std::ofstream out(out_csv);
    out << "subject_id,realization,model,MC\n";
    for (auto& rows : buf) for (auto& r : rows)
        out << r.sid << "," << r.rep << "," << r.model << "," << r.mc << "\n";
    std::cerr << "wrote " << out_csv << "\n";
    return 0;
}
