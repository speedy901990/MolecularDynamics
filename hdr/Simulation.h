#ifndef SIMULATION_H
#define SIMULATION_H

#include "Structure.h"
using namespace std;

class Simulation {
 public:
  enum Potential{ LENARD_JONES };
  Potential potentialType;
  
  static Simulation * instance();
  int perform(Structure * structure);
  int init(string fileName, int argc, char ** argv);
  int checkStructure(Structure * structure);

 private: 
  static Simulation * pInstance;
  static bool initCompleted;

  Simulation();
  Simulation(Simulation const&);
  void operator=(Simulation  const&);
  int loadConfigFromFile(string fileName);
  int checkPotentialType(int potential);
};

#endif /* SIMULATION_H */
