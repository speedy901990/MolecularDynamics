#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#include "Global.h"
#include "Simulation.h"
#include "Structure.h"
#include "GpuKernel.h"
#include "GpuDisplay.h"

class GpuHandler {
 public:
  static GpuHandler * instance();
  int init(int argc, char ** argv);
  
  GpuKernel kernel;

 private:
  int argc;
  char ** argv;
  int * devicesID;
  int devicesCount;

  GpuHandler();
  static GpuHandler * pInstance;
  void operator=(Simulation const&);
  void displayUsageInfo();
  int areParamsInitialized();
  int parseInputParams();
};

#endif /* GPU_HANDLER_H */
