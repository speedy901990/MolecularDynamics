#ifndef PERFORMANCESTATISTICS_H
#define PERFORMANCESTATISTICS_H

#include <cuda_runtime.h>

class PerformanceStatistics {
 public:
  float msecTotal;
  int nIter;
  dim3 block;
  dim3 grid;

  PerformanceStatistics(float msecTotal, int nIter, dim3 block, dim3 grid);
};

#endif // PERFORMANCESTATISTICS_H
