#include "Atom.h"

struct Dimensions {
  int x, y, z;
};

class Structure {
 public:
  Structure();
  Atom * atoms;
  Dimensions dim;
  int atomsCount;
  float force;
  int forceType;

  int init();
};


