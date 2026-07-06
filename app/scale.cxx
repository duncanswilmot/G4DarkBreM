/**
 * @file scale.cxx
 * definition of g4db-scale executable
 */

#include <fstream>
#include <iostream>

#include "G4DarkBreM/G4APrime.h"
#include "G4DarkBreM/G4DarkBreMModel.h"
#include "G4Electron.hh"
#include "G4MuonMinus.hh"

/**
 * printout how to use g4db-scale
 */
void usage() {
  std::cout
      << "USAGE:\n"
         "  g4db-scale [options] db-lib\n"
         "\n"
         "Run the scaling procedure for the input incident energy and madgraph "
         "file\n"
         "\n"
         "This executable is a low-level way to directly test the scaling "
         "procedure implemented\n"
         "inside the G4DarkBreMModel without cluttering the results with the "
         "rest of the Geant4\n"
         "simulation machinery. This means a better understanding of how the "
         "model functions is\n"
         "necessary to be able to effectively use this program.\n"
         " - The 'incident energy' input here is the energy of the lepton JUST "
         "BEFORE it dark brems.\n"
         " - The scaling procedure should scale from a MG sample at an energy "
         "ABOVE the incident energy\n"
         " - The scaling procedure generates the recoil lepton's kinematics "
         "assuming the incident\n"
         "   lepton is traveling along the z-axis. The user is expected to "
         "rotate to the actual incident\n"
         "   frame and calculate the outgoing dark photon kinematics assuming "
         "conservation of momentum.\n"
         "\n"
         "ARGUMENTS\n"
         "  db-lib : dark brem event library to load and sample\n"
         "\n"
         "OPTIONS\n"
         "  -h,--help             : produce this help and exit\n"
         "  -o,--output           : output file to write scaled events to\n"
         "  -E,--incident-energy  : energy of incident lepton in GeV\n"
         "  -Z,--target-Z         : atomic Z of target nucleus to scale to\n"
         "  -N,--num-events       : number of events to sample and scale\n"
         "  -M,--ap-mass          : mass of dark photon in MeV\n"
         "  -l,--aprime-id        : A prime lhe id\n"
         "  --muons               : pass to set lepton to muons (otherwise "
         "electrons)\n"
         "  --scale-APrime        : pass to scale the APrime kinematics\n"
         "  --correct-forward     : pass to approximately restore backscattering\n"
      << std::flush;
}

/**
 * definition of g4db-scale
 *
 * We only need to configure the G4DarkBreMModel so
 * we simply define G4APrime and then construct the model
 * so we can call G4DarkBreMModel::scale for the input
 * number of events.
 */
int main(int argc, char* argv[]) try {
  std::string output_filename{"scaled.csv"};
  double incident_energy{4};
  double target_Z{74.0};
  int num_events{10};
  std::string db_lib;
  double ap_mass{0.1};
  int aprime_id{1023};
  bool muons{false};
  bool scale_APrime{false};
  bool correct_forward{false};
  for (int i_arg{1}; i_arg < argc; i_arg++) {
    std::string arg{argv[i_arg]};
    if (arg == "-h" or arg == "--help") {
      usage();
      return 0;
    } else if (arg == "--muons") {
      muons = true;
    } else if (arg == "-o" or arg == "--output") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      output_filename = argv[++i_arg];
    } else if (arg == "-E" or arg == "--incident-energy") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      incident_energy = std::stod(argv[++i_arg]);
    } else if (arg == "-Z" or arg == "--target-Z") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      target_Z = std::stod(argv[++i_arg]);
    } else if (arg == "-M" or arg == "--ap-mass") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      ap_mass = std::stod(argv[++i_arg]);
    } else if (arg == "-l" or arg == "--aprime-id") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      aprime_id = std::stod(argv[++i_arg]);
    } else if (arg == "-N" or arg == "--num-events") {
      if (i_arg + 1 >= argc) {
        std::cerr << arg << " requires an argument after it" << std::endl;
        return 1;
      }
      num_events = std::stoi(argv[++i_arg]);
    } else if (arg == "--scale-APrime") {
      scale_APrime = true;
    } else if (arg == "--correct-forward") {
      correct_forward = true;
    } else if (not arg.empty() and arg[0] == '-') {
      std::cerr << arg << " is not a recognized option" << std::endl;
      return 1;
    } else {
      db_lib = arg;
    }
  }

  if (db_lib.empty()) {
    std::cerr << "ERROR: DB event library not provided." << std::endl;
    return 1;
  }

  double lepton_mass;
  if (muons) {
    lepton_mass = G4MuonMinus::MuonMinus()->GetPDGMass() / GeV;
  } else {
    lepton_mass = G4Electron::Electron()->GetPDGMass() / GeV;
  }

  // the process accesses the A' mass from the G4 particle
  G4APrime::Initialize(ap_mass);

  // create the model, this is where the LHE file is parsed
  //    into an in-memory library to sample and scale from
  g4db::G4DarkBreMModel db_model(
      db_lib, muons,
      0.0,  // threshold
      1.0,  // epsilon
      g4db::G4DarkBreMModel::ScalingMethod::ForwardOnly,
      g4db::G4DarkBreMModel::XsecMethod::Auto,
      50.0,  // max_R_for_full
      aprime_id,   // aprime_lhe_id
      true,  // load_library
      scale_APrime,
      correct_forward); 
  db_model.PrintInfo();
  printf("   %-16s %f\n", "Lepton Mass [MeV]:", lepton_mass * GeV / MeV);
  printf("   %-16s %f\n", "A' Mass [MeV]:", ap_mass / MeV);

  double lepton_mass_squared{lepton_mass * lepton_mass};
  double ap_mass_squared{ap_mass * ap_mass};

  std::ofstream f{output_filename};
  if (not f.is_open()) {
    std::cerr << "Unable to open output file for writing." << std::endl;
    return -1;
  }
  f << "target_Z,incident_energy,recoil_energy,recoil_px,recoil_py,recoil_pz,"
    << "centerMomentum_energy,centerMomentum_px,centerMomentum_py,"
    << "centerMomentum_pz\n";

  for (int i_event{0}; i_event < num_events; ++i_event) {
    std::pair<G4ThreeVector, G4ThreeVector> momenta =
        db_model.scale(target_Z, incident_energy, lepton_mass);
    G4ThreeVector recoil = momenta.first;
    double recoil_energy = sqrt(recoil.mag2() + lepton_mass_squared);
    G4ThreeVector aprime = momenta.second;
    double aprime_energy = sqrt(aprime.mag2() + ap_mass_squared);

    // convert to GeV to match MG output
    f << target_Z << ',' << incident_energy << ',' << recoil_energy / GeV << ','
      << recoil.x() / GeV << ',' << recoil.y() / GeV << ',' << recoil.z() / GeV
      << "," << (recoil_energy + aprime_energy) / GeV << ","
      << (recoil.x() + aprime.x()) / GeV << ","
      << (recoil.y() + aprime.y()) / GeV << ","
      << (recoil.z() + aprime.z()) / GeV << '\n';
  }

  f.close();
  return 0;
} catch (const std::exception& e) {
  std::cerr << "ERROR: " << e.what() << std::endl;
  return 127;
}
