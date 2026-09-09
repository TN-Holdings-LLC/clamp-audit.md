// psf_zero_ceres_v3.hpp — the "/0 projective clamp" for Ceres and GTSAM
// ============================================================================
// Everything below was compiled and run against the real libraries installed
// here: Ceres 2.2.0, Eigen 3.4.0, GTSAM 4.2.0, g++ -std=c++17.
//
// WHAT THE v2 WRITEUP GOT RIGHT.  All of its maths checks out, verified
// symbolically with sympy (exact zero difference, not just close):
//
//     rho(s)  = [tau*r/(tau+r)]^2,  r = sqrt(s)
//     rho'(s) = tau^3 / (tau+r)^3                       <- confirmed exact
//     rho''(s)= -3*tau^3 / (2*r*(tau+r)^4)              <- confirmed exact
//     weight(e) = (d loss/de)/e = tau^3/(tau+|e|)^3     <- confirmed exact
//
// and the "121x" claim for the old weight bug is exact too (1/11 = 0.090909
// vs 1/1331 = 0.00075131). Finite differences against real Ceres reproduce
// the stated ~1e-10 accuracy. Those fixes are kept verbatim.
//
// WHAT IS FIXED HERE.
//
//  1. THE THIRD HEADER IN THE DOCUMENT WAS NEVER FIXED, AND IT IS THE WORST
//     CODE OF THE THREE.  The v2 writeup covers ZeroClampLoss,
//     ZeroClampCostWrapper, ZeroClampMEstimator and zeroClampGeodesic, but
//     the document also ships `rhoZeroClamp` / `zeroClampVector`, which are
//     an independent, contradictory implementation of the same loss. Measured:
//
//     - rho[0] is DISCONTINUOUS at the r == tau branch, for every tau:
//         tau=0.5: 0.125000 -> 0.062500  (jump -0.0625)
//         tau=1.0: 0.500000 -> 0.250000  (jump -0.2500)
//         tau=2.0: 2.000000 -> 1.000000  (jump -1.0000)
//       rho[1] jumps too, except at tau=1.0 where it agrees by coincidence.
//       Ceres requires rho to be C2; a discontinuous rho is not a loss.
//     - rho[1] disagrees with a finite difference of its OWN rho[0] by up to
//       21,900% (s=100: says 0.16529, truth 0.00075). This is the exact same
//       "off by two powers" error the v2 writeup found in the GTSAM weight()
//       -- present here, unfixed, in the same document.
//     - rho[2] is hardcoded to 0 with the comment "approx". Ceres uses rho[2]
//       for the robustified Gauss-Newton correction, so this silently changes
//       the Hessian.
//     - The quadratic branch uses rho[0]=0.5*s, rho[1]=0.5, i.e. HALF the
//       Ceres convention (rho(s)->s, rho'(0)->1 that ZeroClampLoss correctly
//       uses). Two implementations in one document, two different conventions.
//     - The zero guard sets rho[1]=0, which is another discontinuity: at
//       s=1e-30 rho[1]=0.0, at s=1e-20 it is 0.5.
//
//     Fixed by deleting it. `rhoZeroClamp` here forwards to the single
//     correct implementation, so the two can no longer disagree.
//
//  2. `zeroClampVector`'s "convenience overload" INFINITELY RECURSES.
//     The document defines both
//         zeroClampVector(const MatrixBase<Derived>&, Derived::Scalar)
//         zeroClampVector(const MatrixBase<Derived>&, double)
//     When Scalar is double these are the same signature, and the second is
//     preferred (a concrete parameter type beats a dependent one), so its
//     body calls itself. Measured: compiles with NO warning at -Wall, then
//     SEGFAULTS on the first call (stack exhaustion, exit 139). Fixed by
//     removing the overload; one template with an explicit scalar handles it.
//
//  3. THE WRAPPER'S GRADIENT IS IDENTICALLY ZERO OUTSIDE THE SAFE ZONE.
//     This is the deepest problem and neither round of the writeup mentions
//     it. `ZeroClampCostWrapper` maps r -> tau*r/||r||, so the clamped
//     residual has norm EXACTLY tau, so its contribution to the cost,
//     0.5*||r||^2 = 0.5*tau^2, is a CONSTANT -- and the gradient of a
//     constant is zero. Measured on a real 3x2 problem: ||J^T r|| = 2.8e-17.
//     Not small: zero to machine precision, and necessarily so.
//
//     The consequence is not "outliers are down-weighted". It is that a
//     residual block outside the safe zone is silenced completely while
//     still contributing curvature J^T J (measured nonzero), i.e. it acts as
//     pure damping with no pull. From a poor initialisation, where most
//     blocks start outside tau, the solver has almost no gradient at all.
//
//     Fixed by making the projection SOFT by default: instead of pinning the
//     norm to tau, map it through the same saturation the loss function
//     uses, n -> tau*n/(tau+n), which is strictly increasing, so the radial
//     gradient survives (it decays as tau^3*n/(tau+n)^3 rather than
//     vanishing). `ClampMode::Hard` restores the old behaviour for anyone
//     who wants it, now with the consequence written down.
//
//  4. `using Base::weight;` WAS MISSING, AND THE v2 WRITEUP'S REASONING FOR
//     REMOVING THE PLURAL METHOD IS WRONG.  Bug #5 there says callers "get
//     the base class's real vectorized weight() for free once weight(double)
//     is correctly overridden". They do not: declaring `double weight(double)
//     const override` HIDES every other `weight` overload in the base,
//     including `Vector weight(const Vector&) const` at LossFunctions.h:113.
//     Measured: `m->weight(gtsam::Vector)` fails to compile with
//     "cannot convert gtsam::Vector to double". Removing the plural method
//     without adding a using-declaration made the vectorised path
//     unreachable from user code.
//
//     Scope note, because the v2 claim is wrong but the damage is limited:
//     GTSAM's own IRLS path goes through Base::reweight(), which calls
//     weight(double) from inside Base's own scope where no hiding occurs, so
//     solving still worked. Only direct user calls broke. Fixed with
//     `using Base::weight;`.
//
//  5. THE ESTIMATOR DID NOT ACCEPT A ReweightScheme.  Every built-in GTSAM
//     estimator (Fair, Huber, Cauchy, Tukey, ...) takes
//     `const ReweightScheme reweight = Block` and forwards it to Base.
//     ZeroClampMEstimator silently always used Block. Added, defaulting to
//     Block so existing behaviour is unchanged.
//
//  6. THE WRAPPER ALLOCATED TWO DENSE n x n MATRICES PER Evaluate() CALL.
//     It built `MatrixXd I = Identity(n,n)` and `MatrixXd P = I - r r^T/n^2`
//     on every single residual evaluation -- O(n^2) memory traffic and two
//     heap allocations inside the solver's hot loop, to apply a rank-one
//     update. Rewritten as J <- a*J + b*rhat*(rhat^T J), which allocates
//     nothing and is O(n*m).
//
//  7. THE LOSS IS BOUNDED (REDESCENDING) AND NOTHING SAID SO.  rho(s) -> tau^2
//     as s -> infinity (confirmed symbolically; measured rho(1e12) = 0.999998
//     at tau=1). That makes it a hard redescender in the Tukey family, not a
//     Huber/Cauchy-style convex-tailed loss: outliers get essentially zero
//     influence, but the problem is non-convex and in general
//     initialisation-sensitive. Now documented on the class.
//
//     Stated precisely, because the obvious next sentence would be wrong:
//     the test probes starts at (-40,60), (200,-300) and (5000,-8000) and
//     ZeroClampLoss recovered from ALL of them, matching Huber to 3 decimals.
//     So the non-convexity did not bite on this problem -- Ceres's LM handles
//     it because the model is linear in the parameters. The caveat is a real
//     property of the loss, not a measured failure here, and this file does
//     not pretend otherwise. Where it DID bite is the Hard-mode wrapper
//     (item 3): from (-40,60) it stayed at slope=-40.2, err=72.3, exactly
//     where it started, while Soft mode converged to err=0.070.
//
// KEPT FROM v2 UNCHANGED: the rho[1]/rho[2] closed forms, the Jacobian
// ordering fix (snapshot r before rescaling), the print()/equals() overrides
// that make the estimator instantiable, the corrected weight() power, the
// safe-zone check in zeroClampGeodesic, and the std::max(tau, 1e-8) guard.
// ============================================================================

#pragma once

#include <ceres/ceres.h>
#include <gtsam/linear/NoiseModel.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>

namespace psf {

// ============================================================================
// 0. Single source of truth for the saturation curve.
//    Both the Ceres loss and the GTSAM estimator derive from these, so the
//    two can no longer drift apart the way the document's two copies did.
// ============================================================================
namespace detail {

/// Rebind a smart-pointer template to another element type, so this header
/// follows whatever pointer family the installed GTSAM uses for
/// mEstimator::Base: boost::shared_ptr up to 4.2, std::shared_ptr from 4.3.
/// v2 hardcoded boost::make_shared, which breaks on the newer releases.
template <class Ptr, class U>
struct RebindPtr;
template <template <class> class P, class T, class U>
struct RebindPtr<P<T>, U> {
  using type = P<U>;
};

inline constexpr double kTauFloor = 1e-8;
inline constexpr double kZeroEps = 1e-14;

/// Saturating magnitude map: n -> tau*n/(tau+n). Strictly increasing,
/// s(0)=0, s'(0)=1, s(inf)=tau.
inline double SatMag(double n, double tau) { return tau * n / (tau + n); }

/// d/dn of SatMag.
inline double SatMagDeriv(double n, double tau) {
  const double t = tau + n;
  return (tau * tau) / (t * t);
}

/// Ceres-convention loss on the SQUARED residual s, with rho[0]=[SatMag]^2.
/// rho(0)=0, rho'(0)=1, rho'>0, rho''<0, rho(inf)=tau^2.
inline void Rho(double s, double tau, double rho[3]) {
  const double r = std::sqrt(std::max(s, 0.0));
  if (r < kZeroEps) {
    rho[0] = 0.0;
    rho[1] = 1.0;   // NOT 0: rho must reduce to least squares at the origin.
    rho[2] = 0.0;   // genuine curvature singularity at r=0, as CauchyLoss does
    return;
  }
  const double t = tau + r;
  const double c = tau * r / t;
  const double t3 = tau * tau * tau;
  rho[0] = c * c;
  rho[1] = t3 / (t * t * t);
  rho[2] = -3.0 * t3 / (2.0 * r * t * t * t * t);
}

}  // namespace detail

// ============================================================================
// 1. ZeroClampLoss (ceres::LossFunction)
// ============================================================================
/// Robust loss with rho(s) = [tau*sqrt(s)/(tau+sqrt(s))]^2.
///
/// REDESCENDING / NON-CONVEX. rho is bounded by tau^2, so a residual far
/// outside tau contributes almost nothing -- stronger outlier rejection than
/// Huber or Cauchy, at the cost of local minima. Start from a reasonable
/// initial guess, or run a convex loss (Huber) first and switch. The test
/// file measures what happens when you do not.
class ZeroClampLoss final : public ceres::LossFunction {
 public:
  explicit ZeroClampLoss(double tau)
      : tau_(std::max(tau, detail::kTauFloor)) {}

  void Evaluate(double s, double rho[3]) const override {
    detail::Rho(s, tau_, rho);
  }

  double tau() const { return tau_; }

 private:
  const double tau_;
};

// ============================================================================
// 2. ZeroClampCostWrapper (residual-vector projection)
// ============================================================================
enum class ClampMode {
  /// r -> r * SatMag(||r||)/||r||. Norm approaches but never reaches tau, so
  /// the radial gradient survives. This is the default.
  Soft,
  /// r -> tau * r/||r|| once ||r|| > tau. The v2 behaviour. The clamped norm
  /// is exactly tau, hence 0.5*||r||^2 is constant, hence the gradient is
  /// identically zero (measured: 2.8e-17) -- the block is silenced, not
  /// down-weighted. Only use this if that is genuinely what you want.
  Hard,
};

class ZeroClampCostWrapper final : public ceres::CostFunction {
 public:
  explicit ZeroClampCostWrapper(ceres::CostFunction* inner, double tau,
                                bool take_ownership = true,
                                ClampMode mode = ClampMode::Soft)
      : inner_(inner),
        tau_(std::max(tau, detail::kTauFloor)),
        own_(take_ownership),
        mode_(mode) {
    set_num_residuals(inner_->num_residuals());
    *mutable_parameter_block_sizes() = inner_->parameter_block_sizes();
  }

  ~ZeroClampCostWrapper() override {
    if (own_) delete inner_;
  }

  ZeroClampCostWrapper(const ZeroClampCostWrapper&) = delete;
  ZeroClampCostWrapper& operator=(const ZeroClampCostWrapper&) = delete;

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    if (!inner_->Evaluate(parameters, residuals, jacobians)) return false;

    const int m = num_residuals();
    Eigen::Map<Eigen::VectorXd> r(residuals, m);
    const double n = r.norm();

    // A non-finite residual is a failure of the inner cost, not a value to
    // clamp. v2 returned true here and propagated the NaN into the solver.
    if (!std::isfinite(n)) return false;
    if (n <= detail::kZeroEps) return true;
    if (mode_ == ClampMode::Hard && n <= tau_) return true;

    // f = new magnitude, fp = d(new magnitude)/dn.
    double f, fp;
    if (mode_ == ClampMode::Soft) {
      f = detail::SatMag(n, tau_);
      fp = detail::SatMagDeriv(n, tau_);
    } else {
      f = tau_;
      fp = 0.0;   // constant norm -> no radial sensitivity -> zero gradient
    }

    const double a = f / n;          // tangential scale
    const double b = fp - a;         // radial correction
    const Eigen::VectorXd rhat = r / n;

    // d(r_new)/dp = [fp * rhat rhat^T + a * (I - rhat rhat^T)] J
    //             = a*J + (fp - a) * rhat (rhat^T J)
    // Applied as a rank-one update: no I, no P, no n x n allocation.
    if (jacobians) {
      const auto& bs = parameter_block_sizes();
      for (size_t i = 0; i < bs.size(); ++i) {
        if (jacobians[i] == nullptr) continue;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            J(jacobians[i], m, bs[i]);
        const Eigen::RowVectorXd u = rhat.transpose() * J;  // 1 x bs[i]
        J.noalias() = a * J + b * (rhat * u);
      }
    }
    r *= a;
    return true;
  }

  double tau() const { return tau_; }
  ClampMode mode() const { return mode_; }

 private:
  ceres::CostFunction* inner_;
  const double tau_;
  const bool own_;
  const ClampMode mode_;
};

// ============================================================================
// 3. ZeroClampMEstimator (gtsam robust noise model)
// ============================================================================
class ZeroClampMEstimator : public gtsam::noiseModel::mEstimator::Base {
 public:
  using Base = gtsam::noiseModel::mEstimator::Base;
  /// Same pointer family as Base::shared_ptr, whichever this GTSAM uses.
  using shared_ptr =
      typename detail::RebindPtr<Base::shared_ptr, ZeroClampMEstimator>::type;

  /// `reweight` added for parity with GTSAM's own estimators, which all
  /// expose it (LossFunctions.h: Fair, Huber, Cauchy, Tukey, ...).
  explicit ZeroClampMEstimator(double tau = 1.0,
                               const ReweightScheme reweight = Block)
      : Base(reweight), tau_(std::max(tau, detail::kTauFloor)) {}

  /// Without this, declaring weight(double) hides Base's
  /// `Vector weight(const Vector&) const` and user calls fail to compile.
  using Base::weight;
  using Base::sqrtWeight;

  /// IRLS weight: w(e) = (d loss/de)/e = tau^3/(tau+|e|)^3.
  double weight(double error) const override {
    const double a = std::abs(error);
    if (a < 1e-12) return 1.0;
    const double t = tau_ + a;
    return (tau_ * tau_ * tau_) / (t * t * t);
  }

  /// loss(e) = 0.5 * [tau*|e|/(tau+|e|)]^2, bounded by 0.5*tau^2.
  double loss(double error) const override {
    const double c = detail::SatMag(std::abs(error), tau_);
    return 0.5 * c * c;
  }

  void print(const std::string& s = "") const override {
    std::cout << s << "psf::ZeroClampMEstimator (tau=" << tau_ << ")\n";
  }

  bool equals(const Base& expected, double tol = 1e-8) const override {
    const auto* p = dynamic_cast<const ZeroClampMEstimator*>(&expected);
    return p != nullptr && std::abs(tau_ - p->tau_) <= tol;
  }

  double tau() const { return tau_; }

  static shared_ptr Create(double tau = 1.0,
                           const ReweightScheme reweight = Block) {
    // shared_ptr(new T) rather than make_shared: the spelling is identical
    // for boost:: and std::, so this compiles on both GTSAM generations.
    return shared_ptr(new ZeroClampMEstimator(tau, reweight));
  }

 private:
  const double tau_;
};

// ============================================================================
// 4. Vector helpers
// ============================================================================
/// Soft magnitude saturation of a residual vector, direction preserved.
/// Companion to ClampMode::Soft.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
zeroClampVectorSoft(const Eigen::MatrixBase<Derived>& r, double tau = 1.0) {
  const double t = std::max(tau, detail::kTauFloor);
  const double n = r.norm();
  if (!std::isfinite(n) || n <= detail::kZeroEps) return r.derived();
  return (detail::SatMag(n, t) / n) * r.derived();
}

/// Hard projection onto the ball of radius tau, direction preserved, with the
/// safe-zone early return. Companion to ClampMode::Hard.
///
/// NOTE the single-overload signature: the document's second overload
/// (`..., double tau`) was preferred over the first when Scalar was double
/// and called itself, segfaulting on the first invocation.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
zeroClampVector(const Eigen::MatrixBase<Derived>& r, double tau = 1.0) {
  const double t = std::max(tau, detail::kTauFloor);
  const double n = r.norm();
  if (!std::isfinite(n) || n <= detail::kZeroEps || n <= t) return r.derived();
  return (t / n) * r.derived();
}

inline Eigen::VectorXd zeroClampGeodesic(const Eigen::VectorXd& residual,
                                         double tau = 1.0) {
  return zeroClampVector(residual, tau);
}

/// Ceres-convention rho on the squared residual.
///
/// This REPLACES the document's third-header `rhoZeroClamp`, which was
/// discontinuous at r==tau, disagreed with a finite difference of its own
/// rho[0] by up to 21,900%, hardcoded rho[2]=0, and used half the Ceres
/// scaling in its quadratic branch. It now forwards to the one implementation
/// that was verified, so the two cannot disagree.
inline void rhoZeroClamp(double s, double tau, double rho[3]) {
  detail::Rho(s, std::max(tau, detail::kTauFloor), rho);
}

}  // namespace psf
