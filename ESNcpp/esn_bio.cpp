// esn_bio — MC with global input vs thalamic input vs a random-input-pair null,
// on a small subject subset. Output feeds the {MC, MC_Bio, MC_BioNull} boxplots.
#include "esn.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.txt";
    Config cfg(cfg_path);

    ESNParams base;
    base.N=cfg.geti("reservoir_size",90); base.steps=cfg.geti("steps",6000);
    base.washout=cfg.geti("washout",100); base.n_win=cfg.geti("win_reps",1);
    base.rho=cfg.getd("spectral_radius",0.99); base.ridge=cfg.getd("ridge",1e-4);
    base.train_ratio=cfg.getd("train_ratio",0.7); base.tau_max=cfg.geti("tau",20);
    base.seed=(unsigned)cfg.geti("seed",42);

    std::vector<int> thal = cfg.getints("thal_nodes");   // e.g. 76,77
    if (thal.empty()) thal = {76, 77};
    int n_rand_pairs = cfg.geti("n_rand_pairs", 50);

    std::string in_csv  = cfg.gets("in_csv",  "data/connectomes_sub.csv");
    std::string out_csv = cfg.gets("out_csv", "bio_results.csv");
    int threads = cfg.geti("threads", 0);
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::vector<int> ids; std::vector<MatrixXd> mats;
    read_connectomes(in_csv, base.N, ids, mats);
    const int S = (int)mats.size();
    std::cerr << "esn_bio: " << S << " subjects | thal=";
    for (int t : thal) std::cerr << t << " ";
    std::cerr << "| n_rand_pairs=" << n_rand_pairs << "\n";

    std::vector<std::array<double,3>> res(S);   // MC, MC_Bio, MC_BioNull
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; ++i) {
        std::mt19937 rng(base.seed + 1000u * (unsigned)ids[i]);

        ESNParams pg = base;                       // global input
        double mc_glob = mc_total(mats[i], pg, rng);

        ESNParams pb = base; pb.input_nodes = thal; // thalamic input
        double mc_bio = mc_total(mats[i], pb, rng);

        // random-input-pair null: average MC over n_rand_pairs random node pairs
        std::uniform_int_distribution<int> pick(0, base.N - 1);
        double acc = 0; int got = 0;
        for (int r = 0; r < n_rand_pairs; ++r) {
            int a = pick(rng), b = pick(rng); int guard = 0;
            while (b == a && guard++ < 10) b = pick(rng);
            ESNParams pr = base; pr.input_nodes = {a, b};
            acc += mc_total(mats[i], pr, rng); ++got;
        }
        double mc_null = got ? acc / got : 0.0;
        res[i] = {mc_glob, mc_bio, mc_null};
        #pragma omp critical
        std::cerr << "  subject " << (i + 1) << "/" << S << "\r";
    }
    std::cerr << "\n";

    std::ofstream out(out_csv);
    out << "subject_id,MC,MC_Bio,MC_BioNull\n";
    for (int i = 0; i < S; ++i)
        out << ids[i] << "," << res[i][0] << "," << res[i][1] << "," << res[i][2] << "\n";
    std::cerr << "wrote " << out_csv << "\n";
    return 0;
}
