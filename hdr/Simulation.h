#ifndef SIMULATION_H
#define SIMULATION_H

#include "Structure.h"
using namespace std;

class Simulation {
 public:
  enum Potential{ LENARD_JONES };
  Potential potentialType;
  
  static Simulation * instance();
  int perform();
  int init();
  int loadStructure(Structure * structure);

 private: 
  Simulation();
  Simulation(Simulation const&);
  void operator=(Simulation  const&);
  static Simulation * pInstance;
  int loadConfigFromFile(string fileName);
  int checkPotentialType(int potential);
};

#endif /* SIMULATION_H */
