// ============================================================================
//  esn.hpp — minimal Echo-State-Network core for the connectome MC study
//  Header-only. Each program (esn_mc, esn_surrogates, esn_bio, esn_ipc,
//  esn_trace) includes this file and is compiled independently.
//
//  Conventions (kept identical across all programs and the notebook):
//    * reservoir update : r(t) = tanh( Win u(t) + W r(t-1) ),  r(0)=0
//    * W is the connectome, diagonal zeroed, rescaled so spectral radius = rho
//    * input u(t) ~ Uniform(-1, 1)   (also the natural domain for Legendre IPC)
//    * readout : ridge regression with a bias column
//    * score   : MC(tau) = r^2 between the test-set target u(t-tau) and its
//                linear reconstruction  (Damicelli's r^2, NOT Suarez's |r|)
//    * MC      = sum_{tau=1..tau_max} r^2(tau), averaged over n_win input
//                projections Win
//  All randomness derives from a per-call seed so runs are reproducible and
//  the outer subject loop can be parallelised with OpenMP.
// ============================================================================
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ----------------------------------------------------------------------------
//  Config : a tiny "key = value" file parser (one source of truth = notebook)
// ----------------------------------------------------------------------------
struct Config {
    std::map<std::string, std::string> kv;
    explicit Config(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open config: " + path);
        std::string line;
        while (std::getline(f, line)) {
            auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
            auto eq = line.find('='); if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            auto trim = [](std::string& s){
                s.erase(0, s.find_first_not_of(" \t\r\n"));
                auto p = s.find_last_not_of(" \t\r\n");
                if (p != std::string::npos) s.erase(p + 1); else s.clear(); };
            trim(k); trim(v);
            if (!k.empty()) kv[k] = v;
        }
    }
    std::string gets(const std::string& k, const std::string& d="") const {
        auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    int    geti(const std::string& k, int d)    const { auto it=kv.find(k); return it==kv.end()?d:std::stoi(it->second); }
    double getd(const std::string& k, double d) const { auto it=kv.find(k); return it==kv.end()?d:std::stod(it->second); }
    bool   getb(const std::string& k, bool d)   const { auto it=kv.find(k); if(it==kv.end())return d;
                 std::string v=it->second; return v=="1"||v=="true"||v=="True"||v=="yes"; }
    std::vector<int> getints(const std::string& k) const {
        std::vector<int> out; auto it=kv.find(k); if(it==kv.end()||it->second.empty()) return out;
        std::stringstream ss(it->second); std::string t;
        while (std::getline(ss, t, ',')) { try { out.push_back(std::stoi(t)); } catch(...){} }
        return out; }
};

// ----------------------------------------------------------------------------
//  IO : read connectomes CSV  [subject_id, w0, w1, ... w_{N*N-1}]  (+ header)
//       row-major flattening; diagonal is zeroed on load.
// ----------------------------------------------------------------------------
inline void read_connectomes(const std::string& path, int N,
                             std::vector<int>& ids, std::vector<MatrixXd>& mats) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open connectomes: " + path);
    std::string line;
    std::getline(f, line);                          // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string cell;
        std::getline(ss, cell, ','); ids.push_back(std::stoi(cell));
        MatrixXd A(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) { std::getline(ss, cell, ','); A(i, j) = std::stod(cell); }
        A.diagonal().setZero();
        mats.push_back(std::move(A));
    }
}

// ----------------------------------------------------------------------------
//  ESN parameters
// ----------------------------------------------------------------------------
struct ESNParams {
    int    N           = 90;
    int    steps       = 6000;
    int    washout     = 100;
    int    n_win       = 1;       // input-projection (Win) realisations, averaged
    double rho         = 0.99;    // target spectral radius
    double ridge       = 1e-4;
    double train_ratio = 0.7;
    int    tau_max     = 20;
    unsigned seed      = 42;
    double input_scale = 1.0;     // multiplies every Win entry; 1.0 = original U(-1,1) input
    std::vector<int> input_nodes; // empty => global input to all nodes
};

// Zero diagonal, rescale so that the spectral radius equals rho.
inline MatrixXd rescale_spectral(MatrixXd W, double rho) {
    W.diagonal().setZero();
    Eigen::EigenSolver<MatrixXd> es(W, /*computeEigenvectors=*/false);
    double sr = es.eigenvalues().cwiseAbs().maxCoeff();
    if (sr > 0) W *= (rho / sr);
    return W;
}

// One driven trajectory. Returns the (steps x N) state matrix (r(0)=0 row included).
inline MatrixXd run_states(const MatrixXd& W, const VectorXd& Win, const VectorXd& u) {
    const int T = (int)u.size(), N = (int)W.rows();
    MatrixXd X(T, N);
    VectorXd r = VectorXd::Zero(N);
    X.row(0) = r.transpose();
    for (int t = 1; t < T; ++t) {
        r = (W * r + Win * u(t)).array().tanh();
        X.row(t) = r.transpose();
    }
    return X;
}

// Ridge readout fit on (Ztr,ytr); return r^2 between yte and its prediction on Zte.
inline double r2_ridge(const MatrixXd& Ztr, const VectorXd& ytr,
                       const MatrixXd& Zte, const VectorXd& yte, double ridge) {
    const int p = (int)Ztr.cols();
    MatrixXd A = Ztr.transpose() * Ztr + ridge * MatrixXd::Identity(p, p);
    VectorXd w = A.ldlt().solve(Ztr.transpose() * ytr);
    VectorXd yhat = Zte * w;
    double vy = (yte.array() - yte.mean()).matrix().squaredNorm();
    double vh = (yhat.array() - yhat.mean()).matrix().squaredNorm();
    if (vy <= 0 || vh <= 0) return 0.0;
    double cov = ((yte.array() - yte.mean()) * (yhat.array() - yhat.mean())).sum();
    double r = cov / std::sqrt(vy * vh);
    return r * r;                                   // r^2  (Damicelli)
}

// Build a random input projection Win (global, or restricted to input_nodes).
// Entries are input_scale * U(-1,1); the RNG stream does not depend on input_scale.
inline VectorXd make_Win(const ESNParams& p, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    VectorXd Win = VectorXd::Zero(p.N);
    if (p.input_nodes.empty()) for (int i = 0; i < p.N; ++i) Win(i) = p.input_scale * U(rng);
    else for (int n : p.input_nodes) if (n >= 0 && n < p.N) Win(n) = p.input_scale * U(rng);
    return Win;
}

// Per-delay memory capacity curve r^2(tau), tau=1..tau_max, averaged over n_win.
// If state_r2 is given, it receives the mean squared reservoir state <r_i(t)^2>
// over post-washout steps and nodes (averaged over n_win): the linearity diagnostic.
inline VectorXd mc_curve(const MatrixXd& W, const ESNParams& p, std::mt19937& rng,
                         double* state_r2 = nullptr) {
    MatrixXd Wn = rescale_spectral(W, p.rho);
    VectorXd acc = VectorXd::Zero(p.tau_max);
    double r2_acc = 0.0;
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int rep = 0; rep < p.n_win; ++rep) {
        VectorXd u(p.steps); for (int t = 0; t < p.steps; ++t) u(t) = U(rng);
        VectorXd Win = make_Win(p, rng);
        MatrixXd X = run_states(Wn, Win, u);
        if (state_r2) {
            const int nrow = p.steps - p.washout;
            if (nrow > 0) r2_acc += X.bottomRows(nrow).squaredNorm() / (double(nrow) * p.N);
        }

        if (p.washout >= p.tau_max) {
            // fast path: design matrix identical for all tau -> factor Z^T Z once.
            int t0 = p.washout, n = p.steps - t0;
            if (n < 40) continue;
            MatrixXd Z(n, p.N + 1);
            for (int k = 0; k < n; ++k) { Z(k,0)=1.0; Z.block(k,1,1,p.N)=X.row(t0+k); }
            int ntr = std::min(std::max((int)std::lround(p.train_ratio*n),10), n-10);
            MatrixXd Ztr = Z.topRows(ntr), Zte = Z.bottomRows(n-ntr);
            MatrixXd A = Ztr.transpose()*Ztr + p.ridge*MatrixXd::Identity(p.N+1,p.N+1);
            Eigen::LDLT<MatrixXd> ldlt(A);                 // <-- factor ONCE
            MatrixXd ZtrT = Ztr.transpose();
            for (int tau = 1; tau <= p.tau_max; ++tau) {
                VectorXd ytr(ntr), yte(n-ntr);
                for (int k=0;k<ntr;++k)   ytr(k)=u(t0+k-tau);
                for (int k=0;k<n-ntr;++k) yte(k)=u(t0+ntr+k-tau);
                VectorXd w = ldlt.solve(ZtrT*ytr);          // <-- reuse factor
                VectorXd yhat = Zte*w;
                double vy=(yte.array()-yte.mean()).matrix().squaredNorm();
                double vh=(yhat.array()-yhat.mean()).matrix().squaredNorm();
                if (vy>0 && vh>0){ double cov=((yte.array()-yte.mean())*(yhat.array()-yhat.mean())).sum();
                    double r=cov/std::sqrt(vy*vh); acc(tau-1)+=r*r; }
            }
        } else {
            // safe fallback: original per-tau fit (identical to current code)
            for (int tau = 1; tau <= p.tau_max; ++tau) {
                int t0 = std::max(p.washout, tau), n = p.steps - t0;
                if (n < 20) continue;
                MatrixXd Z(n, p.N+1); VectorXd y(n);
                for (int k=0;k<n;++k){ Z(k,0)=1.0; Z.block(k,1,1,p.N)=X.row(t0+k); y(k)=u(t0+k-tau); }
                int ntr = std::min(std::max((int)std::lround(p.train_ratio*n),10), n-10);
                acc(tau-1) += r2_ridge(Z.topRows(ntr), y.head(ntr),
                                       Z.bottomRows(n-ntr), y.tail(n-ntr), p.ridge);
            }
        }
    }
    if (state_r2) *state_r2 = r2_acc / std::max(1, p.n_win);
    return acc / std::max(1, p.n_win);
}

inline double mc_total(const MatrixXd& W, const ESNParams& p, std::mt19937& rng,
                       double* state_r2 = nullptr) {
    return mc_curve(W, p, rng, state_r2).sum();
}

// ----------------------------------------------------------------------------
//  Node-level weight surrogates. All keep the binary topology, the degrees and the
//  symmetry of the (undirected) dMRI connectome, so (W^2)_ii = s_i^2 Y_i holds for every
//  null and the walk expansion applies to them as to the real network.
//    H0  uniform     : w_ij = (s_i/k_i + s_j/k_j)/2       flattens heterogeneity
//    H1  reshuffle   : the real edge weights, reassigned to other edges so that each
//                      node keeps s_i and sum_j w_ij^2 (hence Y_i): same per-node weight
//                      distribution, random fibre -> weight assignment
//    BS  broken-stick: keeps s_i; each node gets the disparity of a random partition
//                      of its strength, E[Y_i] = 2/(k_i+1)
//    SF  sign-flip   : see null_signflip
//  mc_curve() rescales every matrix to spectral radius rho, surrogates included.
//  Directed nulls are avoided on purpose: asymmetry alone gives complex eigenvalues,
//  which raise linear-regime MC by itself (same weights made asymmetric: +3 MC).
// ----------------------------------------------------------------------------
inline MatrixXd null_uniform(const MatrixXd& W) {
    const int n = W.rows();
    VectorXd mean = VectorXd::Zero(n);              // s_i / k_i por nodo
    for (int i = 0; i < n; ++i) {
        double s = 0.0; int k = 0;
        for (int j = 0; j < n; ++j)
            if (W(i, j) > 0) { s += W(i, j); ++k; }
        if (k > 0) mean(i) = s / k;
    }
    MatrixXd M = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (W(i, j) > 0) {
                const double v = 0.5 * (mean(i) + mean(j));
                M(i, j) = v; M(j, i) = v;            // simétrico
            }
    return M;
}
struct UndirectedEdges { std::vector<int> a, b; std::vector<double> w; };

inline UndirectedEdges undirected_edges(const MatrixXd& W) {
    UndirectedEdges E;
    for (int i = 0; i < W.rows(); ++i)
        for (int j = i + 1; j < W.cols(); ++j)
            if (W(i, j) > 0) { E.a.push_back(i); E.b.push_back(j); E.w.push_back(W(i, j)); }
    return E;
}

inline MatrixXd from_edges(const UndirectedEdges& E, int n) {
    MatrixXd M = MatrixXd::Zero(n, n);
    for (size_t e = 0; e < E.w.size(); ++e) { M(E.a[e], E.b[e]) = E.w[e]; M(E.b[e], E.a[e]) = E.w[e]; }
    return M;
}

// Simulated annealing over swaps of two edges' weights (the edge set and the weight
// multiset never change). Drives every node's strength s_i and sum of squared weights
// q_i towards the targets st, qt; cost = sum_i (s_i/st_i - 1)^2 + (q_i/qt_i - 1)^2.
inline void anneal_weight_swaps(UndirectedEdges& E, int n, const VectorXd& st, const VectorXd& qt,
                                std::mt19937& rng, long sweeps = 10000, double T0 = 0.01) {
    const int m = (int)E.w.size();
    if (m < 2) return;
    VectorXd s = VectorXd::Zero(n), q = VectorXd::Zero(n);
    for (int e = 0; e < m; ++e) {
        const double w = E.w[e];
        s(E.a[e]) += w; s(E.b[e]) += w; q(E.a[e]) += w * w; q(E.b[e]) += w * w;
    }
    auto cost = [&](int i, double si, double qi) {
        if (st(i) <= 0) return 0.0;
        const double x = si / st(i) - 1, y = qi / qt(i) - 1;
        return x * x + y * y;
    };
    double C = 0;
    for (int i = 0; i < n; ++i) C += cost(i, s(i), q(i));
    std::uniform_int_distribution<int> pick(0, m - 1);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const long steps = sweeps * (long)m;
    const double cool = std::pow(1e-8, 1.0 / steps);
    double T = T0 * C / n;
    for (long t = 0; t < steps; ++t, T *= cool) {
        const int e = pick(rng), f = pick(rng);
        const double we = E.w[e], wf = E.w[f];
        if (e == f || we == wf) continue;
        const int nd[4] = {E.a[e], E.b[e], E.a[f], E.b[f]};
        const double ds[4] = {wf - we, wf - we, we - wf, we - wf};
        const double dq[4] = {wf * wf - we * we, wf * wf - we * we, we * we - wf * wf, we * we - wf * wf};
        double dC = 0; bool merged[4] = {false, false, false, false};
        for (int k = 0; k < 4; ++k) {                 // edges e and f may share an endpoint
            if (merged[k]) continue;
            double DS = ds[k], DQ = dq[k];
            for (int l = k + 1; l < 4; ++l) if (nd[l] == nd[k]) { DS += ds[l]; DQ += dq[l]; merged[l] = true; }
            dC += cost(nd[k], s(nd[k]) + DS, q(nd[k]) + DQ) - cost(nd[k], s(nd[k]), q(nd[k]));
        }
        if (dC <= 0 || U(rng) < std::exp(-dC / T)) {
            for (int k = 0; k < 4; ++k) { s(nd[k]) += ds[k]; q(nd[k]) += dq[k]; }
            E.w[e] = wf; E.w[f] = we; C += dC;
        }
    }
}

// Symmetric scaling M_ij -> a_i a_j M_ij so that every row sums to s0_i.
inline MatrixXd scale_to_strengths(const MatrixXd& M, const VectorXd& s0) {
    VectorXd a = VectorXd::Ones(M.rows());
    for (int it = 0; it < 5000; ++it) {
        const VectorXd r = M * a; double err = 0;
        for (int i = 0; i < M.rows(); ++i) if (s0(i) > 0) {
            err = std::max(err, std::abs(a(i) * r(i) / s0(i) - 1));
            a(i) = std::sqrt(a(i) * s0(i) / r(i));
        }
        if (err < 1e-10) break;
    }
    return a.asDiagonal() * M * a.asDiagonal();
}

// H1: permute the real weights over the undirected edges, then anneal swaps back to each
// node's own s_i and sum_j w_ij^2. Typical residuals: 0.1% in s_i, 0.3% in Y_i (median).
inline MatrixXd null_reshuffle(const MatrixXd& W, std::mt19937& rng) {
    UndirectedEdges E = undirected_edges(W);
    std::shuffle(E.w.begin(), E.w.end(), rng);
    anneal_weight_swaps(E, W.rows(), W.rowwise().sum(), W.array().square().rowwise().sum(), rng);
    return from_edges(E, W.rows());
}

// SF  sign-flip : negate the weights of a random half of the (symmetric) edges.
//     Leaves every |w_ij| — and hence the raw order-2 term (W^2)_ii = sum_j w_ij^2 —
//     unchanged, while randomising the sign of each closed triangle
//     (W^3)_ii = sum_{j,h} w_ij w_jh w_hi. Symmetric (i,j)/(j,i) pairs flip together
//     so W stays symmetric. NB: rho(S∘W) <= rho(W), so the rescaling to rho done in
//     mc_curve() multiplies the order-2 term by (rho(W)/rho(S∘W))^2 >= 1.
inline MatrixXd null_signflip(const MatrixXd& W, std::mt19937& rng) {
    MatrixXd M = W;
    std::bernoulli_distribution flip(0.5);
    for (int i = 0; i < W.rows(); ++i)
        for (int j = i + 1; j < W.cols(); ++j)
            if (W(i, j) != 0.0 && flip(rng)) { M(i, j) = -M(i, j); M(j, i) = -M(j, i); }
    return M;
}

// BS: iid Exp(1) weight per undirected edge (normalised iid exponentials are the
// broken-stick / uniform-spacing law), scaled symmetrically to the real strengths, then
// weight swaps so each node's Y_i matches the broken-stick mean 2/(k_i+1), and rescaled
// to the exact strengths. Result: s_i exact, <k Y / (2k/(k+1))> ~ 1.05 (real ~ 1.6).
inline MatrixXd null_brokenstick(const MatrixXd& W, std::mt19937& rng) {
    const int n = W.rows();
    UndirectedEdges E = undirected_edges(W);
    std::exponential_distribution<double> X(1.0);
    for (auto& w : E.w) w = X(rng);
    const VectorXd s0 = W.rowwise().sum();
    UndirectedEdges F = undirected_edges(scale_to_strengths(from_edges(E, n), s0));
    VectorXd qt(n);
    for (int i = 0; i < n; ++i) {
        const int k = (int)(W.row(i).array() > 0).count();
        qt(i) = k > 0 ? s0(i) * s0(i) * 2.0 / (k + 1) : 0.0;
    }
    anneal_weight_swaps(F, n, s0, qt, rng);
    return scale_to_strengths(from_edges(F, n), s0);
}

// ----------------------------------------------------------------------------
//  Information Processing Capacity — Legendre P1, P2 and the P1*P1 cross term.
//  Targets (orthonormal on the U(-1,1) input):
//    P1(u)            = u                          (linear  -> this sums to MC)
//    P2(u)            = (3u^2 - 1)/2               (quadratic)
//    cross P1*P1      = u(t-a) * u(t-b)            (a<b, memory interaction)
//  Each capacity is the r^2 of the corresponding ridge reconstruction.
// ----------------------------------------------------------------------------
inline VectorXd ipc_terms(const MatrixXd& W, const ESNParams& p, int ipc_tau,
                          bool do_cross, std::mt19937& rng, std::vector<std::string>& names) {
    MatrixXd Wn = rescale_spectral(W, p.rho);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    VectorXd u(p.steps); for (int t = 0; t < p.steps; ++t) u(t) = U(rng);
    VectorXd Win = make_Win(p, rng);
    MatrixXd X = run_states(Wn, Win, u);
    auto P1 = [](double x){ return x; };
    auto P2 = [](double x){ return 0.5 * (3.0 * x * x - 1.0); };

    int t0 = std::max(p.washout, ipc_tau), n = p.steps - t0;
    int ntr = std::min(std::max((int)std::lround(p.train_ratio * n), 10), n - 10);
    MatrixXd Z(n, p.N + 1);
    for (int k = 0; k < n; ++k) { Z(k, 0) = 1.0; Z.block(k, 1, 1, p.N) = X.row(t0 + k); }
    MatrixXd Ztr = Z.topRows(ntr), Zte = Z.bottomRows(n - ntr);
    auto cap = [&](auto target){
        VectorXd y(n); for (int k = 0; k < n; ++k) y(k) = target(t0 + k);
        return r2_ridge(Ztr, y.head(ntr), Zte, y.tail(n - ntr), p.ridge); };

    std::vector<double> vals;
    double lin = 0, quad = 0; int cnt = 0;
    for (int a = 1; a <= ipc_tau; ++a) { lin  += cap([&](int t){ return P1(u(t - a)); }); ++cnt; }
    for (int a = 1; a <= ipc_tau; ++a)   quad += cap([&](int t){ return P2(u(t - a)); });
    names.push_back("Lin"); vals.push_back(lin);
    names.push_back("Quad"); vals.push_back(quad);
    if (do_cross) {
        double cross = 0;
        for (int a = 1; a <= ipc_tau; ++a)
            for (int b = a + 1; b <= ipc_tau; ++b)
                cross += cap([&](int t){ return P1(u(t - a)) * P1(u(t - b)); });
        names.push_back("Cross11"); vals.push_back(cross);
    }
    VectorXd out(vals.size()); for (size_t i = 0; i < vals.size(); ++i) out(i) = vals[i];
    return out;
}
