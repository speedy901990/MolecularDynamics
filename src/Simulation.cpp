#include <iostream>
#include <fstream>
#include <cstdlib>
#include "Simulation.h"
#include "GpuHandler.h"

Simulation::Simulation() {}

Simulation * Simulation::pInstance = NULL;

Simulation * Simulation::instance() {
  if (!pInstance)
    pInstance = new Simulation();
  
  return pInstance;
}

int Simulation::perform(int argc, char ** argv) {
  GpuHandler::instance()->init(argc, argv);

  cout << "Simulation done!" << endl;
}

int Simulation::init(string fileName) {
  if (loadConfigFromFile(fileName) != 0)
    return -1;
  
  return 0;
}

int Simulation::loadConfigFromFile(string fileName) {
  string fullPath = "config/" + fileName;
  ifstream cfgFile(fullPath.c_str());
  if (!cfgFile.is_open()) {
    cfgFile.close();
    return -1;
  }
  string tmp;
  int potential;

  getline(cfgFile, tmp);
  cfgFile >> potential;
  if (checkPotentialType(potential) != 0) {
    cfgFile.close();
    return -1;
  }
  this->potentialType = static_cast<Potential>(potential);
  
  cfgFile.close();
  return 0;
}

int Simulation::checkPotentialType(int potential) {
  for (int i=LENARD_JONES ; i!=LENARD_JONES ; i++)
    if (i == potential)
      return -1;
  return 0;
}
  


int Simulation::loadStructure(Structure * structure) {
  
}
