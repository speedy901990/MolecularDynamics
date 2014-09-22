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

  this->argc = argc;
  this->argv = argv;

  ret = parseInputParams();
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    return ret;
  }

  printf("[Molecular Dynamics Using CUDA] - Initializing...\n\n");
  
  getDevices(devicesID, devicesCount);
  displayChosenDevices(devicesID, devicesCount);

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
    if (devicesCount%2 != 0 && devicesCount != 1) {
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
