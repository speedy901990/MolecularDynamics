#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "Atom.h"

struct Dimensions {
  int x, y, z;
};

class Structure {
 public:
  Structure();
  virtual ~Structure();
  Atom * atoms;
  Dimensions dim;
  int atomsCount;
  float force;
  int forceType;
  enum { TOP, BOTTOM, FRONT, BACK, LEFT, RIGHT, ALL_AROUND }; //forceType

  int init();
};

#endif /* STRUCTURE_H */
