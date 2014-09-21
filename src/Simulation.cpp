#include <iostream>
#include <fstream>
#include <cstdlib>
#include "Simulation.h"
#include "GpuHandler.h"
#include "Structure.h"
#include "Log.h"

Simulation::Simulation() {}

Simulation * Simulation::pInstance = NULL;
bool Simulation::initCompleted = false;

Simulation * Simulation::instance() {
  if (!pInstance)
    pInstance = new Simulation();
  
  return pInstance;
}

int Simulation::perform() {
  int ret = SUCCESS;

  ret = checkStructure(structure);
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  cout << "\n--------------- Computing ----------------" << endl << endl;

  GpuHandler::instance()->kernel.allocateDeviceMemory(structure);
  GpuHandler::instance()->kernel.sendDataToDevice(structure);
  GpuHandler::instance()->kernel.execute(structure, GpuHandler::instance()->isVisualizationOn());
  GpuHandler::instance()->kernel.getDataFromDevice(structure);
  GpuHandler::instance()->kernel.clearDeviceMemory();

  cout << "\n------------- Simulation done! ------------" << endl;
}

int Simulation::init(string fileName, Structure * &structure, int argc, char ** argv) {
  int ret = SUCCESS;

  if (initCompleted)
    return INIT_ALREADY_COMPLETED;

  ret = loadConfigFromFile(fileName);
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__, "FileName: " + fileName);
    exit(EXIT_FAILURE);
  }

  ret = checkStructure(structure);
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    exit(EXIT_FAILURE);
  }

  ret = GpuHandler::instance()->init(argc, argv, structure);
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    exit(EXIT_FAILURE);
  }

  this->structure = structure;

  initCompleted = true;
  return SUCCESS;
}

int Simulation::loadConfigFromFile(string fileName) {
  string fullPath = "config/" + fileName;
  ifstream cfgFile(fullPath.c_str());
  if (!cfgFile.is_open()) {
    cfgFile.close();
    return E_FILE_NOT_FOUND;
  }
  string tmp;
  int potential;

  getline(cfgFile, tmp);
  cfgFile >> potential;
  if (checkPotentialType(potential) != SUCCESS) {
    cfgFile.close();
    return E_CONFIG_FILE_PARSE;
  }
  this->potentialType = static_cast<Potential>(potential);
  
  cfgFile.close();
  return SUCCESS;
}

int Simulation::checkPotentialType(int potential) {
  for (int i=LENARD_JONES ; i!=LENARD_JONES ; i++)
    if (i == potential)
      return FAIL;
  return SUCCESS;
}

int Simulation::checkStructure(Structure * structure) {
  if (structure == NULL || structure->atoms == NULL)
    return E_CORRUPTED_STRUCTURE;

  return SUCCESS;
}
