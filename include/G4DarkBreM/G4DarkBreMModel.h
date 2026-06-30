#ifndef G4DARKBREM_G4DARKBREMMODEL_H
#define G4DARKBREM_G4DARKBREMMODEL_H

#include <map>
#include <memory>
#include <random>

#include "G4DarkBreM/ParseLibrary.h"
#include "G4DarkBreM/PrototypeModel.h"
#include "G4DarkBreM/G4APrime.h"
#include "G4DarkBreM/G4FractionallyCharged.h"
#include "G4DarkBreM/ParseLibrary.h"

// Geant4
#include "G4Electron.hh"
#include "G4EventManager.hh"  //for EventID number
#include "G4MuonMinus.hh"
#include "G4PhaseSpaceDecayChannel.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManager.hh"  //for VerboseLevel
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

// Boost
#include <boost/math/quadrature/gauss_kronrod.hpp>

// STL
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace g4db {

/**
 * @class G4DarkBreMModel
 *
 * Geant4 implementation of the model for a particle undergoing a dark brem
 * where we use an imported event library to decide the outgoing kinematics.
 *
 * This is where all the heavy lifting in terms of calculating cross sections
 * and actually having a lepton do a dark brem occurs. This model depends
 * on several configurable parameters which should be studied for your specific
 * use case in order to better tune this model to your lepton/incident energy/
 * target material situation.
 */
class G4DarkBreMModel : public PrototypeModel {
 public:
  /**
   * @enum ScalingMethod
   *
   * Possible methods to use the dark brem events from the imported library
   * inside of this model.
   */
  enum class ScalingMethod {
    /**
     * Use actual lepton energy and get pT from LHE
     * (such that \f$p_T^2+m_l^2 < E_{acc}^2\f$)
     *
     * Scales the energy so that the fraction of kinetic energy is constant,
     * keeping the \f$p_T\f$ constant.
     *
     * If the \f$p_T\f$ is larger than the new energy, that event
     * is skipped, and a new one is taken from the file. If the loaded library
     * does not fully represent the range of incident energies being seen
     * by the simulation, this will occur frequently.
     *
     * With only the kinetic energy fraction and \f$p_T\f$, the sign of
     * the longitudinal momentum \f$p_z\f$ is undetermined. This method
     * simply chooses the \f$p_z\f$ of the recoil lepton to always be positive.
     */
    ForwardOnly = 1,

    /**
     * Boost LHE vertex momenta to the actual lepton energy
     *
     * The scaling is done via two boosts.
     * 1. Boost out of the center-of-momentum (CoM) frame read in along with the
     *    MadGraph event library.
     * 2. Boost into approximately) the incident lepton energy frame by
     *    constructing a "new" CoM frame using the actual CoM frame's
     *    transverse momentum and lowering the \f$p_z\f$ and energy of
     *    the CoM by the difference between the input incident
     *    energy and the sampled incident energy.
     *
     * After these boosts, the energy of the recoil and its \f$p_T\f$ are
     * extracted.
     */
    CMScaling = 2,
    /**
     * Use LHE vertex as is
     *
     * We simply copy the read-in recoil energy's energy, momentum, and
     * \f$p_T\f$.
     */
    Undefined = 3
  };

  /**
   * @enum XsecMethod
   *
   * Possible methods for calculating the xsec using the WW
   * approximation technique.
   *
   * @see flux_factor_chi_numerical for how the flux factor is calculated
   * Here, we refer to the flux factor as \f$\chi(x,\theta)\f$.
   * @see integrate for how the numerical integrations are performed
   *
   * As the methods are related, they share many variable definitions.
   *
   * \f{equation}{
   * \tilde{u} = -xE_0^2\theta^2 - m_A^2\frac{1-x}{x} - m_\ell^2x
   * \f}
   *
   * - \f$x_{max} = 1 - \max(m_A,m_\ell)/E_0\f$ is the upper limit on the x
   * integration
   * - \f$E_0\f$ is the incoming lepton's energy in GeV
   * - \f$m_\ell\f$ is the lepton's mass in GeV
   * - \f$m_A\f$ is the dark photon's mass in GeV
   * - \f$\alpha_{EW} = 1/137\f$ is the fine-structure constant
   * - \f$\epsilon\f$ is the dark photon mixsing strength with the standard
   * photon
   * - \f$pb/GeV = 3.894\times10^8\f$ is the conversion from GeV to pico-barns
   *
   * The integrand limits for \f$\chi\f$ are calculated as
   * \f{equation}{
   * t_{min} = \left(\frac{\tilde{u}}{2E_0(1-x)}\right)^2 \qquad t_{max} = E_0^2
   * \f}
   */
  enum class XsecMethod {
    /**
     * use the full WW approximation
     *
     * This is the 2D integration of a differential cross section including
     * evaluating the flux factor chi numerically at each sampled x and theta.
     * This effectively means we perform a 3D numerical integration when
     * using this method.
     *
     * It is slow but very faithful to the sampled MadGraph total cross section
     * for high (> 200 GeV) incident energies.
     *
     * \f{equation}{
     * \sigma = \frac{pb}{GeV} \int_0^{x_{max}} \int_0^{0.3} \frac{d\sigma}{dx
     * d\theta} d\theta~dx \f} \f{equation}{ \frac{d\sigma}{dx~d\cos\theta} = 2
     * \alpha_{EW}^3\epsilon^2 \sqrt{x^2E_0^2 - m_A^2}E_0(1-x)
     *     \frac{\chi(x,\theta)}{\tilde{u}^2} \mathcal{A}^2
     * \f}
     * \f{equation}{
     * \mathcal{A}^2 =
     * 2\frac{2-2x+x^2}{1-x}+\frac{4(m_A^2+2m_\ell^2)}{\tilde{u}^2}(\tilde{u}x +
     * m_A^2(1-x) + m_\ell^2x^2) \f}
     */
    Full = 1,

    /**
     * assume \f$\theta=0\f$ in the integrand
     *
     * This reduces one of the dimensions of the numerical integration,
     * allowing the theta integration to be done analytically and
     * so we only have a numerical integration over x and t
     *
     * \f{equation}{
     * \sigma = \frac{pb}{GeV} \int_0^{x_{max}} \chi(x,\theta=0)
     * \frac{d\sigma}{dx} dx \f} \f{equation}{ \frac{d\sigma}{dx}(x) = 4
     * \alpha_{EW}^3\epsilon^2
     * \sqrt{1-\frac{m_A^2}{E_0^2}}\frac{1-x+x^2/3}{m_A^2(1-x)/x+m_\ell^2x} \f}
     */
    Improved = 2,

    /**
     * only calculate the flux factor once
     *
     * This simplifies the numerical integration even further by calculating
     * the flux factor chi only at \f$(x=1, \theta=0)\f$ and then using that
     * value everywhere in the integration of the differential cross section
     * over x.
     *
     * \f{equation}{
     * \sigma = \frac{pb}{GeV} \chi(x=1,\theta=0) \int_0^{x_{max}}
     * \frac{d\sigma}{dx} dx \f}
     *
     * where \f$d\sigma/dx\f$ is the same as Improved.
     * Since this uses the maximum value of \f$\chi\f$ for all values of
     * \f$\chi\f$, we modify the upper limit of integration to avoid making this
     * overestimate too much of an overestimate. Instead of \f$t_{max}=E_0^2\f$
     * we use \f$t_{max}=m_A^2+m_\ell^2\f$.
     */
    HyperImproved = 3,

    /**
     * the default cross section calculation
     *
     * This requires its own enum variant since it is determined
     * by the \f$R = m_A/m_\ell\f$ ratio as configured by other parameters.
     *
     * If \f$R < R_{max}\f$, Full is used while Improved is used otherwise.
     * These boarders were determine qualitatively by looking at a series
     * of cross section plots comparing all of these methods.
     *
     * The parameter \f$R_{max}\f$ is configurable in
     * G4DarkBreMModel::G4DarkBreMModel as max_R_for_full.
     */
    Auto = 4
  };

  /**
   * Set the parameters for this model.
   *
   * @param[in] library_path full path to MG library
   *            The library path is immediately passed to
   *            G4DarkBreMModel::SetMadGraphDataLibrary which simply calls
   *            g4db::parseLibrary (where you should look for the specification
   *            options of a dark brem library).
   * @param[in] muons true if using muons, false for electrons
   * @param[in] threshold minimum energy lepton needs to have to dark brem [GeV]
   *            (overwritten by twice the A' mass if it is less so that it
   *            kinematically makes sense).
   * @param[in] epsilon dark photon mixing strength
   * @param[in] sm G4DarkBreMModel::ScalingMethod on how to scale the events in
   * the library
   * @param[in] xm G4DarkBreMModel::XsecMethod on calculating the total cross
   * section
   * @param[in] max_R_for_full maximum value of R where Full will still be used
   *            **Only used with** G4DarkBreMModel::XsecMethod::Auto (go there
   *            for context)
   * @param[in] aprime_lhe_id PDG ID number for the dark photon in the LHE files
   *            being loaded for the library. **Only used if parsing LHE
   * files.**
   * @param[in] load_library only used in cross section executable where it is
   * known that the library will not be used during program run
   * @param[in] scale_APrime whether to scale the A' kinematics based on the
   * library. If false, define A' momentum to conserve momentum without taking
   * nuclear recoil into account.
   * @param[in] dist_decay_min minimum flight distance [mm] at which to
   * decay the A' if G4APrime::DecayMode is set to FlatDecay
   * @param[in] dist_decay_max maximum flight distance [mm] at which to
   * decay the A' if G4APrime::DecayMode is set to FlatDecay
   * @param[in] decay_particle_id PDG ID of the particle that we decay to
   *           (11 for e+e-, 17 for fcp+fcp-)
   */
  G4DarkBreMModel(const std::string& library_path, bool muons,
                  double threshold = 0.0, double epsilon = 1.0,
                  ScalingMethod sm = ScalingMethod::ForwardOnly,
                  XsecMethod xm = XsecMethod::Auto,
                  double max_R_for_full = 50.0, int aprime_lhe_id = 622,
                  bool load_library = true, bool scale_APrime = false, bool correct_forward = false,
                  double dist_decay_min = 0.0, double dist_decay_max = 1.0, 
                  int decay_particle_id = 11);

  /**
   * Destructor
   */
  virtual ~G4DarkBreMModel() = default;

  /**
   * Print the configuration of this model
   */
  virtual void PrintInfo() const;

  /**
   * Calculates the cross section per atom in GEANT4 internal units.
   *
   * The estimate for the total cross section given the material and the
   * lepton's energy is done using an implementation of the WW approximation
   * using Boost's Math Quadrature library to numerically calculate
   * the integrals.
   *
   * @see XsecMethod for the different methods of calculating the cross section.
   *
   * @param lepton_ke kinetic energy of incoming particle
   * @param atomicZ atomic number of atom
   * @param atomicA atomic mass of atom
   * @return cross section (0. if outside energy cuts)
   */
  virtual G4double ComputeCrossSectionPerAtom(G4double lepton_ke,
                                              G4double atomicA,
                                              G4double atomicZ);

  /**
   * Scale one of the MG events in our library to the input incident
   * lepton energy.
   *
   * This is also helpful for testing the scaling procedure in its own
   * executable separate from the Geant4 infrastructure.
   *
   * @note The vector returned is relative to the incident lepton as if
   * it came in along the z-axis.
   *
   * Gets an energy fraction and transverse momentum (\f$p_T\f$) from the
   * loaded library of MadGraph events using the entry in the library with
   * the nearest incident energy above the actual input incident energy.
   *
   * The scaling of this energy fraction and \f$p_T\f$ to the actual lepton
   * energy depends on the input G4DarkBreMModel::ScalingMethod.
   * In all cases, the azimuthal angle is chosen uniformly between 0 and
   * \f$2\pi\f$.
   *
   * @param[in] target_Z atomic Z of target nucleus
   * @param[in] incident_energy incident total energy of the lepton [GeV]
   * @param[in] lepton_mass mass of incident lepton [GeV]
   * @return G4ThreeVectors representing the outgoing momenta of the recoil
   * lepton and the A', respectively
   */
  std::pair<G4ThreeVector, G4ThreeVector> scale(double target_Z,
                                                double incident_energy,
                                                double lepton_mass);

  /**
   * Simulates the emission of a dark photon + lepton
   *
   * @see scale for how the event library is sampled and scaled to the incident
   * lepton's actual energy
   *
   * After calling scale, we rotate the outgoing lepton's momentum to the
   * frame of the incident particle and then calculate the dark photon's
   * momentum such that three-momentum is conserved.
   *
   * @param[in,out] particleChange structure holding changes to make to particle
   * track
   * @param[in] track current track being processesed
   * @param[in] step current step of the track
   * @param[in] element element we are going to dark brem off of
   */
  virtual void GenerateChange(G4ParticleChange& particleChange,
                              const G4Track& track, const G4Step& step,
                              const G4Element& element);

 private:
  /**
   * Set the library of dark brem events to be scaled.
   *
   * This function loads the library from the on-disk file
   * (or files).
   *
   * @see parseLibrary for how libraries are read and
   * what the structure of a dark brem library is
   *
   * @param path path to the on-disk library
   */
  void SetMadGraphDataLibrary(const std::string& path);

  /**
   * Fill vector of currentDataPoints_ with the same number of items as the
   * madgraph data.
   *
   * Randomly choose a starting point so that the simulation run isn't dependent
   * on the order of the events as written in the LHE library. The random
   * starting position is uniformly chosen using G4Uniform() so. The sample
   * function will loop from the last event parsed back to the first event
   * parsed so that the starting position does not matter.
   *
   * In this function, we also update maxIterations_ so that it is equal to the
   * smallest entry in the library (with a maximum of 10k). This saves time in
   * the situation where an incorrect library was accidentally used and the
   * simulation is looping through events attempting to find one that can fit
   * its criteria.
   */
  void MakePlaceholders();

  /**
   * Returns MadGraph data given an energy [GeV].
   *
   * Gets the energy fraction and \f$p_T\f$ from the imported LHE data.
   * incident_energy should be in GeV, returns the sample outgoing kinematics.
   *
   * Samples from the closest imported incident energy _above_ the given value
   * (this helps avoid biasing issues).
   *
   * @param target_Z atomic Z of target nucleus
   * @param incident_energy energy of particle undergoing dark brem [GeV]
   * @return sample outgoing kinematics
   */
  OutgoingKinematics sample(double target_Z, double incident_energy);

 private:
  /**
   * maximum number of iterations to check before giving up on an event
   *
   * This is only used in the ForwardOnly scaling method and is only
   * reached if the event library energies are not appropriately matched
   * with the energy range of particles that are existing in the simulation.
   */
  unsigned int maxIterations_{10000};

  /**
   * Threshold for non-zero xsec [GeV]
   *
   * Configurable with 'threshold'. At minimum, it is always
   * at least twice the dark photon mass.
   */
  double threshold_;

  /**
   * Epsilon value to plug into xsec calculation
   *
   * @sa ComputeCrossSectionPerAtom for how this is used
   *
   * Configurable with 'epsilon'
   */
  double epsilon_;

  /**
   * PDG ID number for the A' (dark photon) as written in the LHE files
   * being loaded as the dark brem event library.
   *
   * This is only used if parsing LHE files and can be ignored if the
   * library is being loaded from an already-constructed CSV.
   */
  int aprime_lhe_id_;

  /**
   * method to scale events for this model
   */
  ScalingMethod scaling_method_;

  /**
   * method for calculating total cross section
   */
  XsecMethod xsec_method_;

  /**
   * Name of method for persisting into the RunHeader
   */
  std::string method_name_;

  /**
   * Full path to the vertex library used for persisting into the RunHeader
   */
  std::string library_path_;

  /**
   * should we always create a totally new lepton when we dark brem?
   *
   * @note make this configurable? I (Tom E) can't think of a reason NOT to have
   * it... The alternative is to allow Geant4 to decide when to make a new
   * particle by checking if the resulting kinetic energy is below some
   * threshold.
   */
  bool alwaysCreateNewLepton_{true};

  /**
   * whether to scale the outgoing A' momentum based on the MadGraph library.
   *
   * The same scaling method is used as for the recoil lepton.
   * The difference between the azimuthal angle of the A' and the recoil lepton
   * is designed to be preserved from MadGraph.
   * The preservation of this angle and the overall scaling of the A' has only
   * been validated for electrons with incident energies ranging from 1GeV to
   * 10GeV and dark photon masses ranging from 1MeV to 100MeV. While the scaling
   * is expected to be well behaved for muons and at other energy scales, users
   * are encouraged to double-check this behavior in their situation and report
   * issues to the repository. https://github.com/LDMX-Software/G4DarkBreM
   *
   * If false, will define the 3-momentum of the A' to conserve 3-momentum
   * with primary and recoil lepton, not taking into account the nuclear recoil.
   * This is the default because then the user is able to "reconstruct" the true
   * incident lepton's momentum using the recoil lepton and produced dark
   * photon's three-momenta. Scaling the A' momentum is only advisable if the A'
   * momentum _itself_ is important to the search (for example, when the A'
   * decays visibly and the decay is observed in which case the A' momentum
   * deciding where the decay occurs is very important).
   */
  bool scale_APrime_{false};

  /**
   * Whether to correct the outgoing (recoil) lepton momentum to include backward events.
   * Backscatter probability as a function of X = |p_z|/P determined by fited function:
   * probability = 0.5 * std::pow((1 - X), beta) * std::exp(-std::pow(X, gamma)) 
   */

  bool correct_forward_{false};

  /**
   * Minimum flight distance [mm] at which to decay the A'
   * if G4APrime::DecayMode is set to FlatDecay
   */
  double dist_decay_min_{0.0};

  /**
   * Maximum flight distance [mm] at which to decay the A'
   * if G4APrime::DecayMode is set to FlatDecay
   */
  double dist_decay_max_{1.0};

  /**
   * Storage of data from mad graph
   *
   * Maps target Z and incoming lepton energy to various options for outgoing
   * kinematics. This is a hefty map and is what stores **all** of the events
   * imported from the LHE library of dark brem events.
   */
  std::map<int, std::map<double, std::vector<OutgoingKinematics>>>
      madGraphData_;

  /**
   * Stores a map of current access points to mad graph data.
   *
   * Maps target Z and incoming lepton energy to the index of the data vector
   * that we will get the data from.
   *
   * Also sorts the incoming lepton energy so that we can find
   * the sampling energy that is closest above the actual incoming energy.
   */
  std::map<int, std::map<double, unsigned int>> currentDataPoints_;

  /**
   * PDG ID of the particle to decay to if G4APrime::DecayMode is set to
   * FlatDecay
   *
   * Currently only supports 11 (e-) and 17 (fcp-)
   */
  int decay_particle_id_{11};
};

}  // namespace g4db

#endif  // G4DARKBREM_G4DARKBREMMODEL_H
