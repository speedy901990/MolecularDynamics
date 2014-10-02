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
  GpuKernel kernel;

  int init(int argc, char ** argv, Structure * &structure);
  bool isVisualizationOn();
  void processInputStructure();

 private:
  int argc;
  char ** argv;
  int * devicesID;
  int devicesCount;
  Structure *structure;
  
  GpuHandler();
  bool visualization;
  static GpuHandler * pInstance;
  void operator=(Simulation const&);
  void displayUsageInfo();
  int areParamsInitialized();
  int parseInputParams();
  void divideStructureForMultiGpu();
};

#endif /* GPU_HANDLER_H */
