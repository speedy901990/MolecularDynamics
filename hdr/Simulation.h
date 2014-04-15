#ifndef SIMULATION_H
#define SIMULATION_H

#include "Structure.h"

class Simulation {
 public:
  int potentialType;
  
  static Simulation * instance();
  int perform();
  int setInitParams();
  int loadStructure(Structure * structure);
  enum { LENARD_JONES }; //potentialType
  
 private:
  Simulation();
  Simulation(Simulation const&);
  void operator=(Simulation  const&);
  static Simulation * pInstance;
};

#endif /* SIMULATION_H */
