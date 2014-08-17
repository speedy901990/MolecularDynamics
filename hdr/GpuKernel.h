#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include "Global.h"
#include "Structure.h"

struct DevMemory {
  Structure * inputAtomsStructure;
  Structure * outputAtomsStructure;
  //TODO result
};

class GpuKernel {
 public:
  GpuKernel();
  ~GpuKernel();
  int allocateDeviceMemory();
  int sendDataToDevice(Structure * atomsStructure);
  int executeKernel();
  int getDataFromDevice();
  int clearDeviceMemory();

 private:
  DevMemory devicePtr;
};


#endif /* GPU_KERNEL_H */
