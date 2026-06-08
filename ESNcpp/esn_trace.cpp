// esn_trace — for ONE subject, dump the raw ingredients of the MC paradigm so
// the notebook can draw the schematic from real data:
//   trace_input.csv    t,u
//   trace_states.csv   t,x0,x1,...            (a few nodes, a short window)
//   trace_readout.csv  k,y_tauA,yhat_tauA,y_tauB,yhat_tauB   (test set)
//   trace_mccurve.csv  tau,r2
#include "esn.hpp"
#include <iostream>

// Local ridge fit that also returns the test-set prediction (needed for the
// readout scatter); mirrors r2_ridge in esn.hpp.
static void fit_predict(const MatrixXd& Ztr, const VectorXd& ytr,
                        const MatrixXd& Zte, const VectorXd& yte,
                        double ridge, VectorXd& yhat, double& r2) {
    int p = (int)Ztr.cols();
    MatrixXd A = Ztr.transpose() * Ztr + ridge * MatrixXd::Identity(p, p);
    VectorXd w = A.ldlt().solve(Ztr.transpose() * ytr);
    yhat = Zte * w;
    double vy = (yte.array() - yte.mean()).matrix().squaredNorm();
    double vh = (yhat.array() - yhat.mean()).matrix().squaredNorm();
    double cov = ((yte.array() - yte.mean()) * (yhat.array() - yhat.mean())).sum();
    r2 = (vy > 0 && vh > 0) ? (cov * cov) / (vy * vh) : 0.0;
}

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.txt";
    Config cfg(cfg_path);

    ESNParams p;
    p.N=cfg.geti("reservoir_size",90); p.steps=cfg.geti("steps",6000);
    p.washout=cfg.geti("washout",100); p.n_win=1;
    p.rho=cfg.getd("spectral_radius",0.99); p.ridge=cfg.getd("ridge",1e-4);
    p.train_ratio=cfg.getd("train_ratio",0.7); p.tau_max=cfg.geti("tau",20);
    p.seed=(unsigned)cfg.geti("seed",42);

    int win   = cfg.geti("trace_window", 200);          // timesteps to dump for input/states
    int n_node = cfg.geti("trace_nodes", 6);            // state traces to dump
    std::vector<int> rtaus = cfg.getints("trace_taus"); // readout delays to dump
    if (rtaus.size() < 2) rtaus = {1, 10};

    std::string in_csv = cfg.gets("in_csv", "data/connectome_one.csv");
    std::string odir   = cfg.gets("trace_dir", ".");

    std::vector<int> ids; std::vector<MatrixXd> mats;
    read_connectomes(in_csv, p.N, ids, mats);
    if (mats.empty()) { std::cerr << "esn_trace: no connectome in " << in_csv << "\n"; return 1; }
    MatrixXd Wn = rescale_spectral(mats[0], p.rho);

    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    VectorXd u(p.steps); for (int t = 0; t < p.steps; ++t) u(t) = U(rng);
    VectorXd Win = make_Win(p, rng);
    MatrixXd X = run_states(Wn, Win, u);

    int w0 = p.washout, w1 = std::min(p.steps, p.washout + win);
    { std::ofstream f(odir + "/trace_input.csv"); f << "t,u\n";
      for (int t = w0; t < w1; ++t) f << (t - w0) << "," << u(t) << "\n"; }
    { std::ofstream f(odir + "/trace_states.csv"); f << "t";
      int nn = std::min(n_node, p.N); for (int j = 0; j < nn; ++j) f << ",x" << j; f << "\n";
      for (int t = w0; t < w1; ++t) { f << (t - w0);
          for (int j = 0; j < nn; ++j) f << "," << X(t, j); f << "\n"; } }

    // readout scatter for two delays + full MC(tau) curve
    VectorXd curve = VectorXd::Zero(p.tau_max);
    std::vector<VectorXd> yte_keep, yhat_keep;
    for (int tau = 1; tau <= p.tau_max; ++tau) {
        int t0 = std::max(p.washout, tau), n = p.steps - t0;
        MatrixXd Z(n, p.N + 1); VectorXd y(n);
        for (int k = 0; k < n; ++k) { Z(k,0)=1.0; Z.block(k,1,1,p.N)=X.row(t0+k); y(k)=u(t0+k-tau); }
        int ntr = std::min(std::max((int)std::lround(p.train_ratio*n),10), n-10);
        VectorXd yhat; double r2;
        fit_predict(Z.topRows(ntr), y.head(ntr), Z.bottomRows(n-ntr), y.tail(n-ntr), p.ridge, yhat, r2);
        curve(tau-1) = r2;
        if (tau == rtaus[0] || tau == rtaus[1]) { yte_keep.push_back(y.tail(n-ntr)); yhat_keep.push_back(yhat); }
    }
    { std::ofstream f(odir + "/trace_mccurve.csv"); f << "tau,r2\n";
      for (int tau = 1; tau <= p.tau_max; ++tau) f << tau << "," << curve(tau-1) << "\n"; }
    { std::ofstream f(odir + "/trace_readout.csv");
      f << "k,y_tauA,yhat_tauA,y_tauB,yhat_tauB\n";
      int m = std::min((int)yte_keep[0].size(), 400);
      for (int k = 0; k < m; ++k)
          f << k << "," << yte_keep[0](k) << "," << yhat_keep[0](k)
                 << "," << yte_keep[1](k) << "," << yhat_keep[1](k) << "\n"; }

    std::cerr << "esn_trace: wrote trace_*.csv (subject_id=" << ids[0]
              << ", MC=" << curve.sum() << ", taus=" << rtaus[0] << "," << rtaus[1] << ")\n";
    return 0;
}
