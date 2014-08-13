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

int GpuHandler::init(int argc, char ** argv) {
  int ret = SUCCESS;

  this->argc = argc;
  this->argv = argv;

  ret = parseInputParams();
  if (ret != SUCCESS) {
    Log::instance()->toConsole(ret, typeid(this).name(), __FUNCTION__, __LINE__);
    return ret;
  }


  printf("[Molecular Dynamics Using CUDA] - Initializing...\n");

  return SUCCESS;
}

int GpuHandler::areParamsInitialized() {
  if (argc <= 1)
    return E_GPU_PARAMS;

  return SUCCESS;
}

int GpuHandler::parseInputParams() {
  if(areParamsInitialized())
    return FAIL;

  if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
    displayUsageInfo();
  }

  return SUCCESS;
}

void GpuHandler::displayUsageInfo() {
  printf("Usage: (* -> required)\n");
  printf("\t-device=n (n >= 0 for deviceID)\n");
  printf("\t-devLimit=devicesCount (Limit of devices used to computing - use 1 or numer of devices)\n");
  printf("*\t-size=size (Atoms Structure size)\n");
  printf("\t-deviceList (Displays devices list)\n");
}
