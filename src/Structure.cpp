#include <fstream>
#include "Structure.h"
#include "Log.h"
bool Structure::initCompleted = false;

Structure::Structure() {
  /*int count = 1;
  dim.x = count;
  dim.y = count;
  dim.z = count;
  atomsCount = count;
  atoms = new Atom[count];*/
}

Structure::~Structure() {
  delete [] atoms;
}

int Structure::init(string fileName) {
  int ret = SUCCESS;
  
  if (initCompleted)
    return INIT_ALREADY_COMPLETED;

  ret = loadConfigFromFile(fileName);
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__, "FileName: " + fileName);
    exit(1);
  }

  atomsCount = dim.x * dim.y * dim.z;

  atoms = new Atom[atomsCount];
  int tmpCount = 0;
  for (int i=0 ; (i<dim.x) && (tmpCount < atomsCount) ; i++) {
    for (int j=0 ; (j<dim.y) && (tmpCount < atomsCount) ; j++) {
      for (int k=0 ; (k<dim.z) && (tmpCount < atomsCount); k++) {
	atoms[tmpCount++].init(i, j, k, 1, Atom::REGULAR);
      }
    }
  }

  initCompleted = true;
  return SUCCESS;
}

int Structure::loadConfigFromFile(string fileName) {
  string fullPath = "config/" + fileName;
  ifstream cfgFile(fullPath.c_str());
  if (!cfgFile.is_open()) {
    cfgFile.close();
    return FAIL;
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
  return SUCCESS;
}

Structure& Structure::operator=(const Structure &orgStruct) {
  if(&orgStruct == this)
    return *this;
  
  delete[] atoms;

  atomsCount = orgStruct.atomsCount;
  atoms = new Atom[atomsCount];
  
  for (int i=0 ; i<orgStruct.atomsCount ; i++) {
    atoms[i] = orgStruct.atoms[i];
  }

  dim.x = orgStruct.dim.x;
  dim.y = orgStruct.dim.y;
  dim.z = orgStruct.dim.z;

  atomsCount = orgStruct.atomsCount;
  force = orgStruct.force;
  forceType = orgStruct.forceType;

  return *this;
}
