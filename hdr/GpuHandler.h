#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#include "Global.h"

class GpuHandler {
 public:
  static GpuHandler * instance();
  int init(int argc, char ** argv);

 private:
  int argc;
  char ** argv;

  GpuHandler();
  static GpuHandler * pInstance;
  void operator=(Simulation const&);
  int displayUsageInfo();
  void isParamsInitialized();
};

#endif /* GPU_HANDLER_H */
