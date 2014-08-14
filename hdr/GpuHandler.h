#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#include "Global.h"
#include "Simulation.h"
#include "Structure.h"

struct DevMemory {
  Structure * inputAtomsStructure;
  Structure * outputAtomsStructure;
  //TODO result
};

class GpuHandler {
 public:
  static GpuHandler * instance();
  int init(int argc, char ** argv);
  int allocateDeviceMemory();
  int sendDataToDevice(Structure * atomsStructure);
  int executeKernel();
  int getDataFromDevice();
  int clearDeviceMemory();

 private:
  int argc;
  char ** argv;
  int * devicesID;
  int devicesCount;
  DevMemory devicePtr;

  GpuHandler();
  static GpuHandler * pInstance;
  void operator=(Simulation const&);
  void displayUsageInfo();
  int areParamsInitialized();
  int parseInputParams();
};

#endif /* GPU_HANDLER_H */
