#include "Structure.h"

class Simulation {
 public:
  int potentialType;
  
  static Simulation * instance();
  int perform();
  int setInitParams();
  int loadStructure(Structure * structure);

 private:
  Simulation();
  static Simulation * pInstance;
};
