#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include "Global.h"
#include "Structure.h"
#include "GpuDisplay.h"

struct DevMemory {
  Structure * inputAtomsStructure;
  Structure * outputAtomsStructure;
  //TODO result
};

class GpuKernel {
  friend class GpuDisplay;
 public:
  GpuKernel();
  ~GpuKernel();
  int allocateDeviceMemory();
  int sendDataToDevice(Structure * atomsStructure);
  int execute(bool displayOn = true);
  int getDataFromDevice();
  int clearDeviceMemory();

 private:
  DevMemory devicePtr;
  int executeDisplayOn();
  int executeDisplayOff();
};


#endif /* GPU_KERNEL_H */
