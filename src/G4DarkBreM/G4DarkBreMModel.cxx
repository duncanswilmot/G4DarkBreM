/**
 * @file G4DarkBreMModel.cxx
 * @brief Source file for dark brem model with some
 * static functions that are helpful to be documented
 */

#include "G4DarkBreM/G4DarkBreMModel.h"

namespace g4db {

/**
 * The integration method we will be using for our numerical integrals
 *
 * The Gauss-Kronrod method was chosen due to its ability to limit the
 * number of calls to the function representing the integrand which
 * should help improve performance for us due to the complexity of our
 * integrand. The order of the GK method was chosen after some
 * experimentation, starting at a high value (61) and then lowering
 * it to achieve better performance while checking the accuracy of
 * the results.
 *
 * As explained in the [Boost GK
 * Docs](https://www.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/gauss_kronrod.html),
 * generally the error estimation technique for this method is
 * overly pessimistic, so we can confidently set the maximum
 * depth low and the desired relative error high compared
 * to other methods. We have followed the examples in the docs
 * where we use max_depth to 5 and relative error to 1e-9.
 *
 * @tparam IntegrandFunction function type whose signature is
 * `double(*)(double)`.
 * @param[in] F function to integrate
 * @param[in] low lower limit of integration
 * @param[in] high upper limit of integration
 */
template <typename IntegrandFunction>
static double integrate(IntegrandFunction F, double low, double high) {
  using int_method = boost::math::quadrature::gauss_kronrod<double, 61>;
  return int_method::integrate(F, low, high, 5, 1e-9);
}

/**
 * numerically integrate the value of the flux factor chi
 *
 * The integration of the form factor into the flux factor can
 * be done analytically with a tool like mathematica, but when
 * including the inelastic term, it produces such a complicated
 * result that the numerical integration is actually *faster*
 * than the analytical one.
 *
 * The form factors are copied from Appendix A (Eq A18 and A19) of
 * https://journals.aps.org/prd/pdf/10.1103/PhysRevD.80.075018
 * and then corrected by Appendix B (Eq B.2 and B.3) of
 * https://arxiv.org/pdf/2401.06843
 * (note that \f$t=Q^2\f$).
 *
 * Here, the equations are listed for reference.
 * \f{equation}{
 * \chi(x,\theta) = \int^{t_{max}}_{t_{min}} dt \left(
 * \frac{Z^2a^4t^2}{(1+a^2t)^2(1+t/d)^2}+\frac{Za_p^4t^2}{(1+a_p^2t)^2(1+t/0.71)^4}\left(1+\frac{t(\mu_p^2-1)}{4m_p^2}\right)\right)\frac{t-t_{min}}{t^2}
 * \f}
 * where
 * \f{equation}{
 * a = \frac{111.0}{m_e Z^{1/3}}
 * \quad
 * a_p = \frac{773.0}{m_e Z^{2/3}}
 * \quad
 * d = \frac{0.164}{A^{2/3}}
 * \f}
 * - \f$m_e\f$ is the mass of the electron in GeV
 * - \f$m_p = 0.938\f$ is the mass of the proton in GeV
 * - \f$\mu_p = 2.79\f$ is the proton \f$\mu\f$
 * - \f$A\f$ is the atomic mass of the target nucleus in amu
 * - \f$Z\f$ is the atomic number of the target nucleus
 *
 * @param[in] A atomic mass of the target nucleus in amu
 * @param[in] Z atomic number of target nucleus
 * @param[in] tmin lower limit of integration over t
 * @param[in] tmax upper limit of integration over t
 */
static double flux_factor_chi_numerical(G4double A, G4double Z, double tmin,
                                        double tmax) {
  /*
   * bin = (mu_p^2 - 1)/(4 m_pr^2)
   * mel = mass of electron in GeV
   */
  static const double bin = (2.79 * 2.79 - 1) / (4 * 0.938 * 0.938),
                      mel = 0.000511;
  const double ael = 111.0 * pow(Z, -1. / 3.) / mel,
               del = 0.164 * pow(A, -2. / 3.),
               ain = 773.0 * pow(Z, -2. / 3.) / mel, din = 0.71,
               ael_inv2 = pow(ael, -2), ain_inv2 = pow(ain, -2);

  /**
   * We've manually expanded the integrand to cancel out the 1/t^2 factor
   * from the differential, this helps the numerical integration converge
   * because we aren't teetering on the edge of division by zero
   *
   * The `auto` used in the integrand definition represents a _function_
   * whose return value is a `double` and which has a single input `lnt`.
   * This lambda expression saves us the time of having to re-calculate
   * the form factor constants that do not depend on `t` because it
   * can inherit their values from the environment.
   * The return value is a double since it is calculated
   * by simple arithmetic operations on doubles.
   *
   * The integrand is so sharply peaked at t close to tmin,
   * it is very helpful to do the integration in the variable
   * u = ln(t) rather than t itself.
   */
  auto integrand = [&](double lnt) {
    double t = exp(lnt);
    double ael_factor = 1. / (ael_inv2 + t), del_factor = 1. / (1 + t / del),
           ain_factor = 1. / (ain_inv2 + t), din_factor = 1. / (1 + t / din),
           nucl = (1 + t * bin);

    // (
    //   (ael_factor * del_factor * Z)^2
    //   + Z * ain_factor^2 * nucl * din_factor^4
    // ) * (t - tmin) * t
    // Following https://arxiv.org/pdf/2401.06843
    return ((ael_factor * del_factor * Z * ael_factor * del_factor * Z) +
            Z * ain_factor * ain_factor * nucl * din_factor * din_factor *
                din_factor * din_factor) *
           (t - tmin) * t;
  };

  return integrate(integrand, log(tmin), log(tmax));
}

G4DarkBreMModel::G4DarkBreMModel(const std::string &library_path, bool muons,
                                 double threshold, double epsilon,
                                 ScalingMethod scaling_method,
                                 XsecMethod xsec_method, double max_R_for_full,
                                 int aprime_lhe_id, bool load_library,
                                 bool scale_APrime, bool correct_forward, 
                                 double dist_decay_min,
                                 double dist_decay_max, int decay_particle_id)
    : PrototypeModel(muons),
      maxIterations_{10000},
      threshold_{std::max(threshold,
                          2. * G4APrime::APrime()->GetPDGMass() / CLHEP::GeV)},
      epsilon_{epsilon},
      aprime_lhe_id_{aprime_lhe_id},
      scaling_method_{scaling_method},
      xsec_method_{xsec_method},
      library_path_{library_path},
      scale_APrime_{scale_APrime},
      correct_forward_{correct_forward},
      dist_decay_min_{dist_decay_min},
      dist_decay_max_{dist_decay_max},
      decay_particle_id_{decay_particle_id} {
  if (xsec_method_ == XsecMethod::Auto) {
    static const double MA = G4APrime::APrime()->GetPDGMass() / GeV;
    const double lepton_mass{(muons_ ? G4MuonMinus::MuonMinus()->GetPDGMass()
                                     : G4Electron::Electron()->GetPDGMass()) /
                             GeV};
    double mass_ratio = MA / lepton_mass;
    if (mass_ratio < max_R_for_full) {
      xsec_method_ = XsecMethod::Full;
    } else {
      xsec_method_ = XsecMethod::Improved;
    }
  }
  if (load_library) SetMadGraphDataLibrary(library_path_);
}

void G4DarkBreMModel::PrintInfo() const {
  G4cout << " Dark Brem Vertex Library Model" << G4endl;
  G4cout << "   Threshold [GeV]: " << threshold_ << G4endl;
  G4cout << "   Epsilon:         " << epsilon_ << G4endl;
  G4cout << "   Scaling Method:  ";
  if (scaling_method_ == ScalingMethod::ForwardOnly) {
    G4cout << "ForwardOnly";
  } else if (scaling_method_ == ScalingMethod::CMScaling) {
    G4cout << "CMScaling";
  } else {
    G4cout << "Undefined";
  }
  G4cout << G4endl;
  G4cout << "   Xsec Method:     ";
  if (xsec_method_ == XsecMethod::Full) {
    G4cout << "Full";
  } else if (xsec_method_ == XsecMethod::Improved) {
    G4cout << "Improved";
  } else {
    G4cout << "HyperImproved";
  }
  G4cout << G4endl;
  G4cout << "   Vertex Library:  " << library_path_ << G4endl;
}

G4double G4DarkBreMModel::ComputeCrossSectionPerAtom(G4double lepton_ke,
                                                     G4double A, G4double Z) {
  static const double MA = G4APrime::APrime()->GetPDGMass() / GeV;
  static const double MA2 = MA * MA;
  static const double alphaEW = 1.0 / 137.0;

  const double lepton_mass{(muons_ ? G4MuonMinus::MuonMinus()->GetPDGMass()
                                   : G4Electron::Electron()->GetPDGMass()) /
                           GeV};
  const double lepton_mass_sq{lepton_mass * lepton_mass};

  // the cross section is zero if the lepton does not have enough
  // energy to create an A'
  // the threshold_ can also be set by the user to a higher value
  // to prevent dark-brem within inaccessible regions of phase
  // space
  if (lepton_ke < keV or lepton_ke < threshold_ * GeV) return 0.;

  // Change energy to GeV.
  double lepton_e = lepton_ke / GeV + lepton_mass;
  double lepton_e_sq = lepton_e * lepton_e;

  // deduce integral bounds
  double xmin = 0;
  double xmax = 1 - std::max(lepton_mass, MA) / lepton_e;

  double integrated_xsec{-1};
  if (xsec_method_ == XsecMethod::Full) {
    /*
     * max recoil angle of A'
     *
     * The wide angle A' are produced at a negligible rate
     * so we enforce a hard-coded cut-off to stay within
     * the small-angle regime.
     *
     * We choose the same cutoff as DMG4.
     */
    double theta_max{0.3};

    /*
     * Differential cross section with respect to x and theta
     *
     * Equation (16) from Appendix A of https://arxiv.org/pdf/2101.12192.pdf
     *
     * This `auto` represents a lambda-expression function, inheriting many
     * pre-calculated constants (like lepton_e and chi) while also calculating
     * the variables dependent on the integration variables. The return value
     * of this function is a double since it is calculated by arithmetic
     * operations on doubles.
     */
    auto diff_cross = [&](double x, double theta) {
      // x is limited above by one value of the mass (xmax = 1 - MA/lepton_e)
      // and so we impose the other half of the kinematic limit that the
      // energy given to the A' (x*lepton_e) must be at least its mass
      if (x * lepton_e < MA) return 0.;

      double theta_sq = theta * theta;
      double x_sq = x * x;

      double utilde =
          -x * lepton_e_sq * theta_sq - MA2 * (1. - x) / x - lepton_mass_sq * x;
      double utilde_sq = utilde * utilde;

      // non-zero theta and non-zero m_l
      double tmin = utilde_sq / (4.0 * lepton_e_sq * (1.0 - x) * (1.0 - x));
      // maximum t kinematically limited to the incident lepton energy
      double tmax = lepton_e_sq;

      /*
       * The chi integrand limits given by
       *
       * Eqs (3.20) and (A6) of
       * https://journals.aps.org/prd/pdf/10.1103/PhysRevD.8.3109
       * OR
       * Eqs (3.2) and (3.6) of
       * https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.46.815
       *
       * to be
       *
       * tmax = m^2(1+l)^2
       * tmin = m^2 tmax / (2*E*x*(1-x))^2
       *
       * where
       *
       *  l = E^2x^2theta^2/m^2
       *  m is mass of dark photon
       *  E is the incident lepton energy
       *
       * were investigated in an attempt to control the numerical integration
       * of chi in the hopes that cutting the integral away from odd places
       * would be able to avoid the funky business. This was not successful,
       * but we are leaving them here in case a typo is found in the future
       * or the search is chosen to resume.
      double el = lepton_e_sq*x_sq*theta_sq/MA2;
      double tmax = MA2*pow(1 + el,2);
      double tmin = MA2*tmax / pow(2*lepton_e*x*(1-x),2);
       */

      // require 0 < tmin < tmax to procede
      if (tmin < 0) return 0.;
      if (tmax < tmin) return 0.;

      /*
       * numerically integrate to calculate chi ourselves
       */
      double chi = flux_factor_chi_numerical(A, Z, tmin, tmax);

      /*
       * Amplitude squared is taken from
       * Equation (17) from Appendix A of https://arxiv.org/pdf/2101.12192.pdf
       * with X = V
       */
      double factor1 = 2.0 * (2.0 - 2. * x + x_sq) / (1. - x);
      double factor2 = 4.0 * (MA2 + 2.0 * lepton_mass_sq) / utilde_sq;
      double factor3 = utilde * x + MA2 * (1. - x) + lepton_mass_sq * x_sq;
      double amplitude_sq = factor1 + factor2 * factor3;

      return 2. * pow(epsilon_, 2.) * pow(alphaEW, 3.) *
             sqrt(x_sq * lepton_e_sq - MA2) * lepton_e * (1. - x) *
             (chi / utilde_sq) * amplitude_sq * sin(theta);
    };

    integrated_xsec = integrate(
        [&](double x) {
          auto theta_integrand = [&](double theta) {
            return diff_cross(x, theta);
          };
          return integrate(theta_integrand, 0., theta_max);
        },
        xmin, xmax);
  } else if (xsec_method_ == XsecMethod::Improved) {
    /*
     * do the theta integral analytically by neglecting
     * all theta terms in the integrand.
     */
    integrated_xsec = integrate(
        [&](double x) {
          // x is limited above by one value of the mass (xmax = 1 -
          // MA/lepton_e) and so we impose the other half of the kinematic limit
          // that the energy given to the A' (x*lepton_e) must be at least its
          // mass
          if (x * lepton_e < MA) return 0.;
          double utilde = -MA2 * (1. - x) / x - lepton_mass_sq * x;
          double utilde_sq = utilde * utilde;
          // non-zero theta and non-zero m_l
          double tmin = utilde_sq / (4.0 * lepton_e_sq * (1.0 - x) * (1.0 - x));
          // maximum t kinematically limited to the incident lepton energy
          double tmax = lepton_e_sq;
          // require 0 < tmin < tmax to procede
          if (tmin < 0) return 0.;
          if (tmax < tmin) return 0.;
          double chi = flux_factor_chi_numerical(A, Z, tmin, tmax);
          double beta = sqrt(1 - MA2 / lepton_e_sq), nume = 1. - x + x * x / 3.,
                 deno = MA2 * (1 - x) / x + lepton_mass_sq;
          return 4 * pow(epsilon_, 2) * pow(alphaEW, 3) * chi * beta * nume /
                 deno;
        },
        xmin, xmax);
  } else if (xsec_method_ == XsecMethod::HyperImproved) {
    /*
     * calculate chi once at x=1, theta=0 and then use it
     * everywhere in the integration over dsigma/dx
     *
     * cut off the integration earlier than the lepton energy squared
     * so that this overestimate isn't too much of an overestimate.
     */
    double chi_hiww = flux_factor_chi_numerical(
        A, Z, MA2 * MA2 / (4 * lepton_e_sq), MA2 + lepton_mass_sq);

    integrated_xsec = integrate(
        [&](double x) {
          // x is limited above by one value of the mass (xmax = 1 -
          // MA/lepton_e) and so we impose the other half of the kinematic limit
          // that the energy given to the A' (x*lepton_e) must be at least its
          // mass
          if (x * lepton_e < MA) return 0.;
          double beta = sqrt(1 - MA2 / lepton_e_sq), nume = 1. - x + x * x / 3.,
                 deno = MA2 * (1 - x) / x + lepton_mass_sq;
          return 4 * pow(epsilon_, 2) * pow(alphaEW, 3) * chi_hiww * beta *
                 nume / deno;
        },
        xmin, xmax);
  } else {
    throw std::runtime_error(
        "Unrecognized XsecMethod, should be set using the enum class.");
  }

  static const G4double GeVtoPb = 3.894E08;

  /*
   * The integrated_xsec should be the correct value, we are just
   * converting it to Geant4's pb units here
   */
  G4double cross = integrated_xsec * GeVtoPb * CLHEP::picobarn;

  if (cross < 0.) return 0.;  // safety check all the math

  return cross;
}

std::pair<G4ThreeVector, G4ThreeVector> G4DarkBreMModel::scale(
    double target_Z, double incident_energy, double lepton_mass) {
  // mass A' in GeV
  static const double MA = G4APrime::APrime()->GetPDGMass() / CLHEP::GeV;

  OutgoingKinematics data = sample(target_Z, incident_energy);
  double energy_factor =
      ((incident_energy - lepton_mass - MA) / (data.E - lepton_mass - MA));
  double EAcc = (data.lepton.e() - lepton_mass) * energy_factor + lepton_mass;
  double Pt = data.lepton.perp();
  double P = sqrt(EAcc * EAcc - lepton_mass * lepton_mass);

  double APrimeE{0.0};
  double APrimePt{0.0};
  double APrimeP{0.0};

  CLHEP::HepLorentzVector ap(data.centerMomentum.px() - data.lepton.px(),
                             data.centerMomentum.py() - data.lepton.py(),
                             data.centerMomentum.pz() - data.lepton.pz(),
                             data.centerMomentum.e() - data.lepton.e());

  if (scaling_method_ == ScalingMethod::ForwardOnly) {
    if (scale_APrime_) {
      APrimeE = (ap.e() - MA) * energy_factor + MA;
      APrimePt = ap.perp();
      APrimeP = sqrt(APrimeE * APrimeE - MA * MA);
    }

    unsigned int i = 0;
    while (Pt * Pt + lepton_mass * lepton_mass > EAcc * EAcc) {
      // Skip events until the transverse energy is less than the total energy.
      i++;
      data = sample(target_Z, incident_energy);
      energy_factor =
          ((incident_energy - lepton_mass - MA) / (data.E - lepton_mass - MA));
      EAcc = (data.lepton.e() - lepton_mass) * energy_factor + lepton_mass;
      Pt = data.lepton.perp();
      P = sqrt(EAcc * EAcc - lepton_mass * lepton_mass);

      if (scale_APrime_) {
        ap.setPx(data.centerMomentum.px() - data.lepton.px());
        ap.setPy(data.centerMomentum.py() - data.lepton.py());
        ap.setPz(data.centerMomentum.pz() - data.lepton.pz());
        ap.setE(data.centerMomentum.e() - data.lepton.e());
        APrimeE = (ap.e() - MA) * energy_factor + MA;
        APrimePt = ap.perp();
        APrimeP = sqrt(APrimeE * APrimeE - MA * MA);
      }

      if (i > maxIterations_) {
        std::cerr
            << "Could not produce a realistic vertex with library energy "
            << data.lepton.e() << " GeV.\n"
            << "Consider expanding your libary of A' vertices to include a "
               "beam energy closer to "
            << incident_energy << " GeV." << std::endl;
        break;
      }
    }
  } else if (scaling_method_ == ScalingMethod::CMScaling) {
    CLHEP::HepLorentzVector el(data.lepton.px(), data.lepton.py(),
                               data.lepton.pz(), data.lepton.e());

    double ediff = data.E - incident_energy;
    CLHEP::HepLorentzVector newcm(
        data.centerMomentum.px(), data.centerMomentum.py(),
        data.centerMomentum.pz() - ediff, data.centerMomentum.e() - ediff);
    el.boost(-1. * data.centerMomentum.boostVector());
    el.boost(newcm.boostVector());

    double newE = (data.lepton.e() - lepton_mass) * energy_factor + lepton_mass;
    el.setE(newE);
    EAcc = el.e();
    Pt = el.perp();
    P = el.vect().mag();

    if (scale_APrime_) {
      double oldE = ap.e();
      ap.boost(-1. * data.centerMomentum.boostVector());
      ap.boost(newcm.boostVector());

      newE = (oldE - MA) * energy_factor + MA;
      ap.setE(newE);
      APrimePt = ap.perp();
      APrimeP = ap.vect().mag();
    }
  } else if (scaling_method_ == ScalingMethod::Undefined) {
    EAcc = data.lepton.e();
    P = sqrt(EAcc * EAcc - lepton_mass * lepton_mass);
    Pt = data.lepton.perp();

    if (scale_APrime_) {
      APrimeE = ap.e();
      APrimeP = sqrt(APrimeE * APrimeE - MA * MA);
      APrimePt = ap.perp();
    }
  } else {
    throw std::runtime_error(
        "Unrecognized ScalingMethod, should be set by using the enum class.");
  }

  // outgoing lepton momentum
  G4double PhiAcc = G4UniformRand() * 2 * pi;
  G4double recoilMag = sqrt(EAcc * EAcc - lepton_mass * lepton_mass) * GeV;
  G4ThreeVector recoil;
  double ThetaAcc = std::asin(Pt / P);
  recoil.set(std::sin(ThetaAcc) * std::cos(PhiAcc),
             std::sin(ThetaAcc) * std::sin(PhiAcc), std::cos(ThetaAcc));
  recoil.setMag(recoilMag);

  // Apply correction to Forward Only scaling to approximate backward-scattered electrons
  // only if correct_forward is passed to the db model
  if (scaling_method_ == ScalingMethod::ForwardOnly && correct_forward_) {
    double X = recoil.z() / recoil.mag();
    const float beta = 1.155;
    const float gamma = 1.630;
    double prob = 0.5 * std::pow((1 - X), beta) * std::exp(-std::pow(X, gamma));
    double num = G4UniformRand();
    bool backward = (prob > num);

    if (backward) {
      recoil.set(recoil.x(), recoil.y(), -recoil.z());
    }
  }

  // outgoing A' momentum
  G4ThreeVector aprime;
  if (scale_APrime_) {
    // make the azimuthal (phi) angle difference the same as in the MG data
    G4double aPrimePhi = PhiAcc + ap.phi() - data.lepton.phi();

    G4double aPrimeMag = sqrt(APrimeE * APrimeE - MA * MA) * GeV;
    double aPrimeTheta = std::asin(APrimePt / APrimeP);
    aprime.set(std::sin(aPrimeTheta) * std::cos(aPrimePhi),
               std::sin(aPrimeTheta) * std::sin(aPrimePhi),
               std::cos(aPrimeTheta));
    aprime.setMag(aPrimeMag);
  } else {
    // define A' momentum to be 3-momentum conserving with the recoil
    double incident_momentum_mag =
        sqrt(incident_energy * incident_energy - lepton_mass * lepton_mass) *
        GeV;
    aprime = G4ThreeVector(0, 0, incident_momentum_mag) - recoil;
  }
  return std::make_pair(recoil, aprime);
} // end of scale

void G4DarkBreMModel::GenerateChange(G4ParticleChange &particleChange,
                                     const G4Track &track, const G4Step &step,
                                     const G4Element &element) {
  // mass of incident lepton
  double Ml = track.GetDefinition()->GetPDGMass() / CLHEP::GeV;

  // convert to energy units in LHE files [GeV]
  G4double incidentEnergy =
      step.GetPostStepPoint()->GetTotalEnergy() / CLHEP::GeV;

  std::pair<G4ThreeVector, G4ThreeVector> outgoingMomenta =
      scale(element.GetZ(), incidentEnergy, Ml);
  outgoingMomenta.first.rotateUz(track.GetMomentumDirection());
  outgoingMomenta.second.rotateUz(track.GetMomentumDirection());
  G4ThreeVector recoilMomentum = outgoingMomenta.first;
  G4ThreeVector darkPhotonMomentum = outgoingMomenta.second;

  G4DynamicParticle *dphoton =
      new G4DynamicParticle(G4APrime::APrime(), darkPhotonMomentum);

  if (G4APrime::getDecayMode() == G4APrime::DecayMode::FlatDecay) {
    double dist_decay =
        G4UniformRand() * (dist_decay_max_ - dist_decay_min_) + dist_decay_min_;
    CLHEP::HepLorentzVector p4 = dphoton->Get4Momentum();
    double tau_decay = dist_decay / p4.beta() / p4.gamma() / c_light;
    dphoton->SetPreAssignedDecayProperTime(tau_decay);

    // set decay products if any
    if (decay_particle_id_ == 17) {
      dphoton->SetPreAssignedDecayProducts(
        (new G4PhaseSpaceDecayChannel("A^1", 1.0, 2, "fcp-", "fcp+"))
            ->DecayIt(G4APrime::APrime()->G4APrime::APrime()->GetPDGMass()));
    } else if (decay_particle_id_ == 11) {
      dphoton->SetPreAssignedDecayProducts(
        (new G4PhaseSpaceDecayChannel("A^1", 1.0, 2, "e-", "e+"))
            ->DecayIt(G4APrime::APrime()->G4APrime::APrime()->GetPDGMass()));
    } else {
      throw std::runtime_error(
          "Unrecognized decay particle ID for flat decay mode.");
    }
  } // end of flat decay mode

  // stop tracking and create new secondary instead of primary
  if (alwaysCreateNewLepton_) {
    /*
     * Create a new lepton in order to make it easier to extract
     * the outgoing sim-level dark brem kinematics.
     */
    G4DynamicParticle *el =
        new G4DynamicParticle(track.GetDefinition(), recoilMomentum);
    particleChange.SetNumberOfSecondaries(2);
    particleChange.AddSecondary(dphoton);
    particleChange.AddSecondary(el);
    particleChange.ProposeTrackStatus(fStopAndKill);
    // continue tracking
  } else {
    /*
     * just have primary lose energy (don't rename to different track ID)
     *
     * This branch is untested and so we are not sure if it works as expected.
     */
    particleChange.SetNumberOfSecondaries(1);
    particleChange.AddSecondary(dphoton);
    particleChange.ProposeMomentumDirection(recoilMomentum.unit());
    double recoil_energy = sqrt(recoilMomentum.mag2() + Ml * Ml);
    // energy of primary recoiling
    G4double finalKE = recoil_energy - Ml;
    particleChange.ProposeEnergy(finalKE);
  }
} // end of GenerateChange

void G4DarkBreMModel::SetMadGraphDataLibrary(const std::string &path) {
  /*
   * print status to user so they know what's happening
   */
  if (GetVerboseLevel() > 0)
    G4cout << "[ G4DarkBreMModel ] : loading event librariy..." << G4endl;

  parseLibrary(path, aprime_lhe_id_, madGraphData_);

  if (madGraphData_.size() == 0) {
    throw std::runtime_error(
        "BadConf : Unable to find any library entries at '" + path +
        "'\n"
        "  The library is either a single CSV file or a directory of LHE "
        "files.\n"
        "  Any individual file can be compressed with `gzip`.\n"
        "  This means the valid extensions are '.lhe', '.lhe.gz', '.csv', and "
        "'.csv.gz'");
  }

  MakePlaceholders();  // Setup the placeholder offsets for getting data.

  if (GetVerboseLevel() > 0) G4cout << "[ G4DarkBreMModel ] : done" << G4endl;

  /*
   * Print out loaded MG library
   */
  if (GetVerboseLevel() > 1) {
    G4cout << "MadGraph Library of Dark Brem Events:\n";
    for (const auto &Z_submap : madGraphData_) {
      G4cout << "  Target Z: " << Z_submap.first << "\n";
      for (const auto &beam_events : Z_submap.second) {
        G4cout << "\t" << beam_events.first << " GeV Beam -> "
               << beam_events.second.size() << " Events\n";
      }
    }
    G4cout << G4endl;
  }

  return;
}

void G4DarkBreMModel::MakePlaceholders() {
  currentDataPoints_.clear();
  maxIterations_ = 10000;
  for (const auto &Z_submap : madGraphData_) {
    for (const auto &beam_events : Z_submap.second) {
      currentDataPoints_[Z_submap.first][beam_events.first] =
          int(G4UniformRand() * beam_events.second.size());
      if (beam_events.second.size() < maxIterations_)
        maxIterations_ = beam_events.second.size();
    }
  }
}

OutgoingKinematics G4DarkBreMModel::sample(double target_Z,
                                           double incident_energy) {
  // Cycle through imported targets and beam energies until the closest one
  // above is found, or the max is reached. We pick a target Z and then a beam
  // energy.
  int samplingZ = 0;
  for (const auto &Z_submap : currentDataPoints_) {
    samplingZ = Z_submap.first;  // move samplingZ up
    // check if we went under the sampling Z
    // this works since the std::map keys are sorted
    if (target_Z <= samplingZ) break;
  }
  double samplingE = 0.;
  for (const auto &beam_index : currentDataPoints_[samplingZ]) {
    samplingE = beam_index.first;  // move samplingE up
    // check if went under the sampling energy
    //  the map is sorted by key, so we can be done right after E0 goes under
    //  samplingE
    if (incident_energy <= samplingE) break;
  }
  // now samplingE is the closest energy above E0 or the maximum energy imported
  // from mad graph

  // Need to loop around if we hit the end, in case our random
  // starting position happens to be late enough in the file
  if (currentDataPoints_.at(samplingZ).at(samplingE) >=
      madGraphData_.at(samplingZ).at(samplingE).size()) {
    currentDataPoints_[samplingZ][samplingE] = 0;
  }

  // increment the current index _after_ getting its entry from
  // the in-memory library
  return madGraphData_.at(samplingZ).at(samplingE).at(
      currentDataPoints_[samplingZ][samplingE]++);
}

}  // namespace g4db
