//#include <ifstream>
#include "Structure.h"

Structure::Structure() {
  int count = 1;
  dim.x = count;
  dim.y = count;
  dim.z = count;
  atomsCount = count;
  atoms = new Atom[count];
}

Structure::~Structure() {
  delete atoms;
}

int Structure::init() {
  // TODO: read from file
  /*
  ifstream cfgFile("structure.cfg");
  if (!cfgFile.is_open()) {
    cfgFile.close();
    return -1;
  }
  string tmp;
  cfgFile >> tmp;
  cfgFile >> dim.x; 
  cfgFile >> tmp;
  cfgFile >> dim.y;
  cfgFile >> tmp; 
  cfgFile >> dim.z;
  */

  int count = 1;
  dim.x = count;
  dim.y = count;
  dim.z = count;
  atomsCount = dim.x * dim.y * dim.z;

  atoms = new Atom[atomsCount];
  for (int i=0 ; i<dim.x ; i++) {
    for (int j=0 ; j<dim.y ; j++) {
      for (int k=0 ; k<dim.z ; k++) {
	atoms[i].init(i, j, k, 1, Atom::REGULAR);
      }
    }
  }

  //cfgFile.close();
  
  return 0;
}

