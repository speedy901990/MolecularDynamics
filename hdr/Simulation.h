#ifndef SIMULATION_H
#define SIMULATION_H

#include "Structure.h"
using namespace std;

class Simulation {
 public:
  enum Potential{ LENARD_JONES };
  Potential potentialType;
  
  static Simulation * instance();
  int perform(int argc, char ** argv);
  int init(string fileName);
  int loadStructure(Structure * structure);

 private: 
  Simulation();
  Simulation(Simulation const&);
  static Simulation * pInstance;
  void operator=(Simulation  const&);
  int loadConfigFromFile(string fileName);
  int checkPotentialType(int potential);
};

#endif /* SIMULATION_H */
