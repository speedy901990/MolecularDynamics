#include <cuda_runtime.h>

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
  this->argc = argc;
  this->argv = argv;
  displayUsageInfo();
  printf("[Molecular Dynamics Using CUDA] - Initializing...\n");

  return 0;
}

void GpuHandler::isParamsInitialized() {
  if (argc <= 1) {
    cerr << "ERROR: GPU params not initialized. Use: ? for help." << endl;
    exit(EXIT_FAILURE);
  }
}

int GpuHandler::displayUsageInfo() {
  isParamsInitialized();
  if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage: (* -> required)\n");
    printf("\t-device=n (n >= 0 for deviceID)\n");
    printf("\t-devLimit=devicesCount (Limit of devices used to computing - use 1 or numer of devices)\n");
    printf("*\t-size=size (Atoms Structure size)\n");
    printf("\t-deviceList (Displays devices list)\n");

    exit(EXIT_SUCCESS);
  }

  return -1;
}
