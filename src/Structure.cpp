#include <fstream>
#include "Structure.h"
using namespace std;

Structure::Structure() {
  int count = 1;
  dim.x = count;
  dim.y = count;
  dim.z = count;
  atomsCount = count;
  atoms = new Atom[count];
}

Structure::~Structure() {
  delete [] atoms;
}

int Structure::init() {
  string fileName = "structure.cfg";
  loadFromFile(fileName);

  atomsCount = dim.x * dim.y * dim.z;

  atoms = new Atom[atomsCount];
  for (int i=0 ; i<dim.x ; i++) {
    for (int j=0 ; j<dim.y ; j++) {
      for (int k=0 ; k<dim.z ; k++) {
	atoms[i].init(i, j, k, 1, Atom::REGULAR);
      }
    }
  }

  return 0;
}
 
int Structure::loadFromFile(string fileName) {
  string fullPath = "config/" + fileName;
  ifstream cfgFile(fullPath.c_str());
  if (!cfgFile.is_open()) {
    cfgFile.close();
    return -1;
  }
  string tmp;

  getline(cfgFile, tmp);
  cfgFile >> dim.x; 
  getline(cfgFile, tmp);
  getline(cfgFile, tmp);
  cfgFile >> dim.y;
  getline(cfgFile, tmp);
  getline(cfgFile, tmp);
  cfgFile >> dim.z;
  
  cfgFile.close();
}

