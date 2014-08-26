#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "Global.h"
#include "Atom.h"
using namespace std;

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
  
  int init(string fileName);
  Structure& operator=(const Structure &orgStruct);

 private:
  static bool initCompleted;

  int loadConfigFromFile(string fileName);
};

#endif /* STRUCTURE_H */
