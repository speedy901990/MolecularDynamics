#include <cuda_runtime.h>

#include "Log.h"
#include "CudaHelpers.h"
#include "GpuHandler.h"
#include "helper_functions.h"

GpuHandler::GpuHandler() {}

GpuHandler * GpuHandler::pInstance = NULL;

GpuHandler * GpuHandler::instance() {
  if (!pInstance)
    pInstance = new GpuHandler();

  return pInstance;
}

int GpuHandler::init(int argc, char ** argv, Structure * &structure) {
  int ret = SUCCESS;
  visualization = true;
  this->structure = structure;

  this->argc = argc;
  this->argv = argv;

  ret = parseInputParams();
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    return ret;
  }

  printf("[Molecular Dynamics Using CUDA] - Initializing...\n\n");
  
  //getDevices(devicesID, devicesCount);
  //displayChosenDevices(devicesID, devicesCount);

  if (visualization)
    GpuDisplay::instance()->init(argc, argv, structure);

  return SUCCESS;
}

int GpuHandler::areParamsInitialized() {
  if (argc <= 1)
    return E_GPU_PARAMS;

   return SUCCESS;
}

int GpuHandler::parseInputParams() {
  if(areParamsInitialized())
    return E_PARAMS_NOT_INITIALIZED;
  
  // help
  if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
    displayUsageInfo();
    exit(EXIT_SUCCESS);
  }

  // deviceList
  if (checkCmdLineFlag(argc, (const char **)argv, "devicesList")) {
    displayAvailableDevices();
    exit(EXIT_SUCCESS);
  }
  
  //device and deviceLimit
  if (checkCmdLineFlag(argc, (const char **)argv, "deviceID")) {
    devicesCount = 1;
    devicesID = new int[devicesCount];
    devicesID[0] = getCmdLineArgumentInt(argc, (const char **)argv, "deviceID");
  }
  else if (checkCmdLineFlag(argc, (const char **)argv, "devicesCount")) {
    devicesCount = getCmdLineArgumentInt(argc, (const char **)argv, "devicesCount");
    if (/*devicesCount%2 != 0 && devicesCount != 1*/false) {
      Log::instance()->toConsole(E_INSUFFICIENT_DEVICES_LIMIT, typeid(this).name(), __FUNCTION__, __LINE__, "use 1 or even number of devices\n");
      exit(EXIT_SUCCESS);
    }
    devicesID = new int[devicesCount];
    for (int i=0 ; i<devicesCount ; i++)
      devicesID[0] = i;
  }
  else {
    Log::instance()->toConsole(I_DEVICE_NOT_SELECTED, typeid(this).name(), __FUNCTION__, __LINE__, "Using default device.");
    devicesCount = 1;
    devicesID = new int[devicesCount];
    devicesID[0] = 0;
    }

  // visualization
  if (checkCmdLineFlag(argc, (const char **)argv, "noVisual")) {
    visualization = false;
  }
  
  return SUCCESS;
}

void GpuHandler::displayUsageInfo() {
  printf("Usage: (* -> required)\n");
  printf("\t-deviceID=n (n >= 0 for deviceID)\n");
  printf("\t-devicesCount=devicesLimit (Number of devices used to computing - use 1 or bigger numer of devices)\n");
  printf("\nMore Info:\n");
  printf("\t-help or ? (Displays this menu)\n");
  printf("\t-devicesList (Displays devices list)\n");
  printf("\t-noVisual (Console mode only)\n");
}

bool GpuHandler::isVisualizationOn() {
  return visualization;
}

void GpuHandler::processInputStructure() {
  if (devicesCount == 1)
    return;
  
  divideStructureForMultiGpu();
}

void GpuHandler::divideStructureForMultiGpu() {
  Structure * multiGpuStruct = new Structure[devicesCount];

  int chunk = structure->dim.x / devicesCount;
  int chunkSize = chunk * structure->dim.y * structure->dim.z;
  int idx = 0;
  int newAtomsIdx = 0;
  int startIdx = 0;
  int endIdx = 0;

  for (int devID=0 ; devID<devicesCount ; devID++) {
    startIdx = devID * chunk * structure->dim.y * structure->dim.z;
    endIdx = (devID + 1) * chunk * structure->dim.y * structure->dim.z;
    newAtomsIdx = 0;
    multiGpuStruct[devID].atoms = new Atom[chunkSize];
    multiGpuStruct[devID].atomsCount = chunkSize;

    //printf("start: %d\tend: %d\n", startIdx, endIdx);

    for (idx = startIdx ; idx<endIdx ; idx++, newAtomsIdx++) {
      multiGpuStruct[devID].atoms[newAtomsIdx] = structure->atoms[idx];
    }
  }

  delete structure;
  structure = multiGpuStruct;

  /*
  for (int devID=0 ; devID<devicesCount ; devID++) {
    printf("Device %d:\n", devID);
    for (int i=0 ; i<structure[devID].atomsCount ; i++) {
      printf("Atom %d:\t\t%f\t%f\t%f\n", i, structure[devID].atoms[i].pos.x, structure[devID].atoms[i].pos.y, structure[devID].atoms[i].pos.z);
    }
  }
  */
}

int GpuHandler::getDevicesCount() {
  return devicesCount;
}
