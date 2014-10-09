#ifndef GPUTHREAD_H
#define GPUTHREAD_H

#include "GpuKernel.h"

class GpuThread {
 public:
  GpuKernel * kernel;
  int tid;

  GpuThread();
  GpuThread(GpuKernel *kernel, int tid);
};

#endif // GPUTHREAD_H
