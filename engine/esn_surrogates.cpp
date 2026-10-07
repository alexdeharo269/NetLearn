// esn_surrogates — MC of the real network vs four symmetric weight nulls, over
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
    p.input_scale = cfg.getd("input_scale", 1.0);      // multiplies Win
    p.ridge *= p.input_scale * p.input_scale;           // states scale with the input: keep ridge relative

    int n_real = cfg.geti("n_real", 10);
    // log_theory: add m2 = Tr(Wn^2)/N = sum_k lambda_k^2 / N and MC_lin (linear theory, same Win)
    // asym_control: add "RealAsym", the real weights made asymmetric. It has its own random
    //               stream, so the rows of the other models do not change.
    const bool log_theory = cfg.getb("log_theory", false);
    const bool asym_control = cfg.getb("asym_control", false);

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
    struct Row { int sid, rep; const char* model; double mc, m2, lin; };
    std::vector<std::vector<Row>> buf(S);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; ++i) {
        std::mt19937 rng(p.seed + 1000u * (unsigned)ids[i]);
        std::mt19937 rng_ctrl(p.seed + 1000u * (unsigned)ids[i] + 7u);
        auto run = [&](int rep, const char* name, const MatrixXd& M, std::mt19937& g) {
            double lin = 0.0, m2 = 0.0;
            const double mc = mc_total(M, p, g, nullptr, log_theory ? &lin : nullptr);
            if (log_theory) { const MatrixXd Wn = rescale_spectral(M, p.rho); m2 = (Wn * Wn).trace() / p.N; }
            buf[i].push_back({ids[i], rep, name, mc, m2, lin});
        };
        for (int rep = 0; rep < n_real; ++rep) {
            // Real: re-seed per realization so Real and nulls see the same input draw stream.
            run(rep, "Real",        mats[i],                       rng);
            run(rep, "BrokenStick", null_brokenstick(mats[i], rng), rng);
            run(rep, "Uniform",     null_uniform(mats[i]),          rng);
            run(rep, "Reshuffle",   null_reshuffle(mats[i], rng),   rng);
            run(rep, "SignFlip",    null_signflip(mats[i], rng),    rng);
            if (asym_control) run(rep, "RealAsym", asymmetric_control(mats[i], rng_ctrl), rng_ctrl);
        }
        #pragma omp critical
        std::cerr << "  subject " << (i + 1) << "/" << S << "\r";
    }
    std::cerr << "\n";

    std::ofstream out(out_csv);
    out << "subject_id,realization,model,MC" << (log_theory ? ",m2,MC_lin" : "") << "\n";
    for (auto& rows : buf) for (auto& r : rows) {
        out << r.sid << "," << r.rep << "," << r.model << "," << r.mc;
        if (log_theory) out << "," << r.m2 << "," << r.lin;
        out << "\n";
    }
    std::cerr << "wrote " << out_csv << "\n";
    return 0;
}
