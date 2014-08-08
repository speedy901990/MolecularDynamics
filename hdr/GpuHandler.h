#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#include <cuda_runtime.h>

#include "Global.h"
//#include "CudaHelpers.h"
#include "helper_functions.h"

class GpuHandler {
 public:
  static GpuHandler * instance();
  int init();

 private:
  GpuHandler();
  static GpuHandler * pInstance;
  void operator=(Simulation const&);
};

#endif /* GPU_HANDLER_H */
