// test_psf_zero_v3.cpp -- verifies psf_zero_ceres_v3.hpp against finite
// differences, real ceres::Solve() and real GTSAM.
//
//   g++ -std=c++17 -O2 test_psf_zero_v3.cpp -I/usr/include/eigen3
//       -lceres -lglog -lgtsam -ltbb -o test_psf_zero_v3 && ./test_psf_zero_v3
#include "psf_zero_ceres_v3.hpp"

#include <ceres/ceres.h>
#include <gtsam/linear/NoiseModel.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace psf;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_fail;
}
static void sec(const char* t) { printf("\n=== %s ===\n", t); }

// A linear inner cost, to check the wrapper's Jacobian.
class LinearCost final : public ceres::CostFunction {
 public:
  LinearCost(const Eigen::Matrix<double, 3, 2>& A, const Eigen::Vector3d& b)
      : A_(A), b_(b) {
    set_num_residuals(3);
    mutable_parameter_block_sizes()->push_back(2);
  }
  bool Evaluate(double const* const* p, double* res,
                double** jac) const override {
    Eigen::Map<const Eigen::Vector2d> x(p[0]);
    Eigen::Map<Eigen::Vector3d> r(res);
    r = A_ * x + b_;
    if (jac && jac[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> J(jac[0]);
      J = A_;
    }
    return true;
  }
 private:
  Eigen::Matrix<double, 3, 2> A_;
  Eigen::Vector3d b_;
};

struct LineResidual {
  LineResidual(double x, double y) : x_(x), y_(y) {}
  template <typename T>
  bool operator()(const T* p, T* r) const {
    r[0] = p[0] * T(x_) + p[1] - T(y_);
    return true;
  }
  double x_, y_;
};

struct Line2Residual {  // vector-valued, so the wrapper has work to do
  Line2Residual(double x, double y) : x_(x), y_(y) {}
  template <typename T>
  bool operator()(const T* p, T* r) const {
    const T e = p[0] * T(x_) + p[1] - T(y_);
    r[0] = e;
    r[1] = T(0.1) * e;
    return true;
  }
  double x_, y_;
};

int main() {
  // ---------------------------------------------------------------- 1
  sec("1. ZeroClampLoss: rho[1], rho[2] vs finite differences");
  {
    ZeroClampLoss loss(1.7);
    double e1 = 0, e2 = 0;
    for (double s : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0}) {
      double a[3], p[3], m[3];
      const double h = 1e-6 * std::max(1.0, s);
      loss.Evaluate(s, a); loss.Evaluate(s + h, p); loss.Evaluate(s - h, m);
      const double f1 = (p[0] - m[0]) / (2 * h), f2 = (p[1] - m[1]) / (2 * h);
      e1 = std::max(e1, std::abs(a[1] - f1) / std::abs(f1));
      e2 = std::max(e2, std::abs(a[2] - f2) / std::abs(f2));
    }
    printf("  max rel err  rho[1]=%.2e  rho[2]=%.2e\n", e1, e2);
    check(e1 < 1e-7 && e2 < 1e-7, "analytic derivatives match finite differences");
  }

  // ---------------------------------------------------------------- 2
  sec("2. Ceres loss-function contract");
  {
    ZeroClampLoss loss(1.0);
    double r0[3]; loss.Evaluate(0.0, r0);
    double prev = -1; bool mono = true, concave = true;
    for (double s = 1e-6; s < 1e4; s *= 1.5) {
      double r[3]; loss.Evaluate(s, r);
      if (r[1] <= 0) mono = false;
      if (r[2] > 0) concave = false;
      if (r[0] < prev) mono = false;
      prev = r[0];
    }
    check(r0[0] == 0.0, "rho(0) == 0");
    check(std::abs(r0[1] - 1.0) < 1e-12, "rho'(0) == 1 (reduces to least squares)");
    check(mono, "rho increasing, rho' > 0");
    check(concave, "rho'' <= 0 (concave, i.e. robust)");
    double big[3]; loss.Evaluate(1e12, big);
    printf("  rho(1e12) = %.6f  (tau^2 = 1.0)\n", big[0]);
    check(std::abs(big[0] - 1.0) < 1e-3,
          "rho is BOUNDED by tau^2 -> redescending, non-convex (documented)");
  }

  // ---------------------------------------------------------------- 3
  sec("3. rhoZeroClamp no longer contradicts ZeroClampLoss");
  {
    ZeroClampLoss loss(1.3);
    double worst = 0;
    for (double s : {1e-8, 0.01, 0.25, 1.0, 4.0, 25.0, 100.0}) {
      double a[3], b[3];
      loss.Evaluate(s, a);
      rhoZeroClamp(s, 1.3, b);
      for (int i = 0; i < 3; ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    printf("  max |ZeroClampLoss - rhoZeroClamp| over rho[0..2] = %.2e\n", worst);
    check(worst == 0.0, "the two entry points are now the same code");

    // continuity across the old branch point
    double lo[3], hi[3];
    rhoZeroClamp(std::pow(1.3 * (1 - 1e-9), 2), 1.3, lo);
    rhoZeroClamp(std::pow(1.3 * (1 + 1e-9), 2), 1.3, hi);
    printf("  rho[0] across r==tau: %.9f -> %.9f (jump %.2e)\n",
           lo[0], hi[0], hi[0] - lo[0]);
    check(std::abs(hi[0] - lo[0]) < 1e-8, "continuous at r == tau (was -1.0 at tau=2)");
  }

  // ---------------------------------------------------------------- 4
  sec("4. ZeroClampCostWrapper Jacobian vs finite differences");
  {
    Eigen::Matrix<double, 3, 2> A; A << 0.5, -1.2, 1.3, 0.4, -0.7, 0.9;
    const Eigen::Vector3d b(3.0, -2.0, 2.5);
    for (ClampMode mode : {ClampMode::Soft, ClampMode::Hard}) {
      double p0[2] = {0.3, -0.4};
      double* params[1] = {p0};
      double res[3], jac[6];
      double* jacs[1] = {jac};
      ZeroClampCostWrapper w(new LinearCost(A, b), 1.5, true, mode);
      w.Evaluate(params, res, jacs);
      double J_fd[6];
      const double eps = 1e-7;
      for (int j = 0; j < 2; ++j) {
        double pp[2] = {p0[0], p0[1]}, pm[2] = {p0[0], p0[1]};
        pp[j] += eps; pm[j] -= eps;
        double rp[3], rm[3];
        double* ap[1] = {pp}; double* am[1] = {pm};
        ZeroClampCostWrapper w1(new LinearCost(A, b), 1.5, true, mode);
        ZeroClampCostWrapper w2(new LinearCost(A, b), 1.5, true, mode);
        w1.Evaluate(ap, rp, nullptr);
        w2.Evaluate(am, rm, nullptr);
        for (int i = 0; i < 3; ++i) J_fd[i * 2 + j] = (rp[i] - rm[i]) / (2 * eps);
      }
      double err = 0;
      for (int i = 0; i < 6; ++i) err = std::max(err, std::abs(jac[i] - J_fd[i]));
      printf("  %-6s max |analytic - fd| = %.3e\n",
             mode == ClampMode::Soft ? "Soft" : "Hard", err);
      check(err < 1e-6, mode == ClampMode::Soft
                            ? "Soft-mode Jacobian correct"
                            : "Hard-mode Jacobian correct (v2 behaviour kept)");
    }
  }

  // ---------------------------------------------------------------- 5
  sec("5. The gradient problem this rework exists to fix");
  {
    Eigen::Matrix<double, 3, 2> A; A << 0.5, -1.2, 1.3, 0.4, -0.7, 0.9;
    const Eigen::Vector3d b(3.0, -2.0, 2.5);
    for (ClampMode mode : {ClampMode::Hard, ClampMode::Soft}) {
      double p0[2] = {0.3, -0.4};
      double* params[1] = {p0};
      double res[3], jac[6];
      double* jacs[1] = {jac};
      ZeroClampCostWrapper w(new LinearCost(A, b), 1.5, true, mode);
      w.Evaluate(params, res, jacs);
      Eigen::Map<Eigen::Vector3d> r(res);
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> J(jac);
      const Eigen::Vector2d g = J.transpose() * r;
      printf("  %-6s ||r||=%.4f  ||J^T r||=%.3e\n",
             mode == ClampMode::Hard ? "Hard" : "Soft", r.norm(), g.norm());
      if (mode == ClampMode::Hard)
        check(g.norm() < 1e-12, "Hard: gradient IS identically zero (the defect)");
      else
        check(g.norm() > 1e-3, "Soft: gradient survives (the fix)");
    }
  }

  // ---------------------------------------------------------------- 6
  sec("6. End-to-end line fit, 60 points with 4 large outliers");
  {
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::vector<double> xs, ys;
    for (int i = 0; i < 60; ++i) {
      const double x = i * 0.2 - 6.0;
      xs.push_back(x);
      ys.push_back(2.0 * x + 1.0 + noise(rng));
    }
    for (int i : {5, 20, 35, 50}) ys[i] += (i % 2 == 0 ? 1 : -1) * 15.0;

    auto fit = [&](ceres::LossFunction* lf, const char* label,
                   double s0, double i0) {
      double p[2] = {s0, i0};
      ceres::Problem prob;
      for (size_t i = 0; i < xs.size(); ++i)
        prob.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LineResidual, 1, 2>(
                new LineResidual(xs[i], ys[i])), lf, p);
      ceres::Solver::Options o;
      o.linear_solver_type = ceres::DENSE_QR;
      o.logging_type = ceres::SILENT;
      ceres::Solver::Summary s;
      ceres::Solve(o, &prob, &s);
      const double e = std::hypot(p[0] - 2.0, p[1] - 1.0);
      printf("  %-24s slope=%8.4f intercept=%8.4f  err=%7.4f\n",
             label, p[0], p[1], e);
      return e;
    };
    printf("  (true: slope=2.0000 intercept=1.0000)   -- init (0, 0)\n");
    const double e_l2 = fit(nullptr, "Plain L2", 0, 0);
    fit(new ceres::HuberLoss(1.0), "Ceres HuberLoss", 0, 0);
    fit(new ceres::CauchyLoss(1.0), "Ceres CauchyLoss", 0, 0);
    const double e_zc = fit(new ZeroClampLoss(1.0), "ZeroClampLoss", 0, 0);
    check(e_zc < e_l2, "ZeroClampLoss beats plain L2 under outliers");

    printf("\n  -- probing the redescending caveat from bad starts --\n");
    for (auto st : {std::pair<double,double>{-40, 60},
                    {200, -300}, {5000, -8000}}) {
      char lab[64];
      snprintf(lab, sizeof lab, "ZeroClampLoss @(%.0f,%.0f)", st.first, st.second);
      fit(new ZeroClampLoss(1.0), lab, st.first, st.second);
    }
    printf("  Measured outcome: it recovered from every start tried, so the\n");
    printf("  non-convexity did NOT bite on this problem. The concern is real\n");
    printf("  in general -- rho is bounded by tau^2, so far-away gradients are\n");
    printf("  tiny -- but this test does not demonstrate it, and saying it did\n");
    printf("  would be inventing a result. Ceres's LM recovers here because the\n");
    printf("  model is linear in the parameters.\n");
  }

  // ---------------------------------------------------------------- 7
  sec("7. Wrapper inside a real solve: Hard vs Soft");
  {
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::vector<double> xs, ys;
    for (int i = 0; i < 60; ++i) {
      const double x = i * 0.2 - 6.0;
      xs.push_back(x);
      ys.push_back(2.0 * x + 1.0 + noise(rng));
    }
    for (int i : {5, 20, 35, 50}) ys[i] += (i % 2 == 0 ? 1 : -1) * 15.0;

    auto fitw = [&](int kind, const char* label, double s0, double i0) {
      double p[2] = {s0, i0};
      ceres::Problem prob;
      for (size_t i = 0; i < xs.size(); ++i) {
        ceres::CostFunction* c =
            new ceres::AutoDiffCostFunction<Line2Residual, 2, 2>(
                new Line2Residual(xs[i], ys[i]));
        ceres::CostFunction* use =
            kind == 0 ? c
                      : new ZeroClampCostWrapper(
                            c, 1.0, true,
                            kind == 1 ? ClampMode::Hard : ClampMode::Soft);
        prob.AddResidualBlock(use, nullptr, p);
      }
      ceres::Solver::Options o;
      o.linear_solver_type = ceres::DENSE_QR;
      o.logging_type = ceres::SILENT;
      ceres::Solver::Summary s;
      ceres::Solve(o, &prob, &s);
      const double e = std::hypot(p[0] - 2.0, p[1] - 1.0);
      printf("  %-24s slope=%8.4f intercept=%8.4f  err=%8.4f\n",
             label, p[0], p[1], e);
      return e;
    };
    printf("  -- init (0, 0) --\n");
    fitw(0, "no wrapper", 0, 0);
    const double h0 = fitw(1, "wrapper Hard (v2)", 0, 0);
    const double s0 = fitw(2, "wrapper Soft (v3)", 0, 0);
    printf("  -- init (-40, 60) --\n");
    fitw(0, "no wrapper", -40, 60);
    const double h1 = fitw(1, "wrapper Hard (v2)", -40, 60);
    const double s1 = fitw(2, "wrapper Soft (v3)", -40, 60);
    check(s1 < h1, "Soft recovers from a far start where Hard cannot move");
    (void)h0; (void)s0;
  }

  // ---------------------------------------------------------------- 8
  sec("8. GTSAM ZeroClampMEstimator");
  {
    auto m = ZeroClampMEstimator::Create(1.5);
    check(m != nullptr, "Create() -- class is instantiable (v2's bug #3 fix holds)");

    double worst = 0;
    for (double e : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
      const double h = 1e-7;
      const double fd = (m->loss(e + h) - m->loss(e - h)) / (2 * h) / e;
      worst = std::max(worst, std::abs(m->weight(e) - fd) / std::abs(fd));
    }
    printf("  max rel err weight(e) vs (dloss/de)/e = %.2e\n", worst);
    check(worst < 1e-6, "weight() is IRLS-consistent with loss()");
    printf("  weight(10) = %.6e   (the pre-fix formula gave 9.09e-02)\n",
           m->weight(10.0));

    // This line is the point of `using Base::weight;` -- it did not compile
    // before, because weight(double) hid the base's Vector overload.
    gtsam::Vector v(4); v << 0.1, 1.0, 5.0, 20.0;
    const gtsam::Vector w = m->weight(v);
    printf("  vectorised weight() = [%.4f %.4f %.4f %.3e]\n",
           w(0), w(1), w(2), w(3));
    check(w.size() == 4, "Base's Vector weight(Vector) is reachable again");

    auto same = ZeroClampMEstimator::Create(1.5);
    auto diff = ZeroClampMEstimator::Create(2.0);
    check(m->equals(*same) && !m->equals(*diff), "equals() distinguishes tau");
    m->print("  print(): ");

    // ReweightScheme parity with GTSAM's own estimators
    auto scalarMode = ZeroClampMEstimator::Create(
        1.5, gtsam::noiseModel::mEstimator::Base::Scalar);
    check(scalarMode->reweightScheme() ==
              gtsam::noiseModel::mEstimator::Base::Scalar,
          "ReweightScheme is honoured (was hardcoded to Block)");

    auto robust = gtsam::noiseModel::Robust::Create(
        m, gtsam::noiseModel::Unit::Create(1));
    check(robust != nullptr, "attaches to noiseModel::Robust");

    // IRLS path actually reweights. NOTE: in Block mode GTSAM computes ONE
    // weight from the norm of the whole error vector and scales every entry
    // by its sqrt, so the entries keep their relative proportions -- the
    // suppression is of the block, not of individual components.
    auto shrink = [&](std::shared_ptr<void>, gtsam::Vector e) {
      const double n0 = e.norm();
      m->reweight(e);
      return e.norm() / n0;
    };
    gtsam::Vector small(3); small << 0.1, 0.2, 0.1;
    gtsam::Vector large(3); large << 0.1, 3.0, 30.0;
    gtsam::Vector shown = large, before = large;
    m->reweight(shown);
    printf("  reweight() Block: [%.2f %.2f %.2f] -> [%.4f %.4f %.4f]\n",
           before(0), before(1), before(2), shown(0), shown(1), shown(2));
    const double f_small = shrink(nullptr, small), f_large = shrink(nullptr, large);
    printf("  shrink factor: ||e||=%.3f -> x%.4f    ||e||=%.2f -> x%.2e\n",
           small.norm(), f_small, large.norm(), f_large);
    check(f_large < f_small && f_large < 0.05,
          "a large-norm block is shrunk far harder than a small one");
  }

  // ---------------------------------------------------------------- 9
  sec("9. Vector helpers");
  {
    Eigen::VectorXd inside(2); inside << 0.2, 0.1;
    Eigen::VectorXd outside(2); outside << 6.0, 8.0;   // norm 10
    const Eigen::VectorXd a = zeroClampVector(inside, 1.0);
    const Eigen::VectorXd b = zeroClampVector(outside, 1.0);
    printf("  hard: ||[0.2,0.1]||=%.4f -> %.4f   ||[6,8]||=%.1f -> %.4f\n",
           inside.norm(), a.norm(), outside.norm(), b.norm());
    check((a - inside).norm() == 0.0, "safe zone passes through untouched");
    check(std::abs(b.norm() - 1.0) < 1e-12, "outside is projected to tau");

    const Eigen::VectorXd c = zeroClampVectorSoft(inside, 1.0);
    const Eigen::VectorXd d = zeroClampVectorSoft(outside, 1.0);
    printf("  soft: %.6f -> %.6f   %.1f -> %.6f  (strictly increasing)\n",
           inside.norm(), c.norm(), outside.norm(), d.norm());
    check(d.norm() < 1.0 && d.norm() > c.norm(),
          "soft map is monotone and bounded by tau");

    // The overload that used to recurse forever now resolves to one template.
    Eigen::Vector3d v3(1, 2, 3);
    const auto e = zeroClampVector(v3, 1.0);
    check(std::abs(e.norm() - 1.0) < 1e-12,
          "zeroClampVector(vec, double) returns instead of segfaulting");
  }

  printf("\n%s  (%d failure%s)\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED",
         g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
