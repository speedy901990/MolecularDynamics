#include "Structure.h"

Structure::Structure() {
  int count = 3;
  dim.x = count;
  dim.y = count;
  dim.z = 1;
  
  atomsCount = dim.x * dim.y * dim.z;
  atoms = new Atom[atomsCount];
  for (int i=0 ; i<atomsCount ; i++) {
    //atoms[i].init(...);
  }
}

int Structure::init() {
  
}

