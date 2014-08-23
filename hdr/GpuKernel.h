#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include "Global.h"
#include "Structure.h"
#include "GpuDisplay.h"

struct DevMemory {
  Structure * inputAtomsStructure;
  Structure * outputAtomsStructure;
  Atom * inputAtoms;
  Atom * outputAtoms;
  //TODO result
};

class GpuKernel {
 public:
  GpuKernel();
  ~GpuKernel();
  int allocateDeviceMemory(Structure * &atomsStructure);
  int sendDataToDevice(Structure * &atomsStructure);
  int execute(bool displayOn = true);
  int getDataFromDevice();
  int clearDeviceMemory();
  int executeDisplayOn();
  int executeDisplayOff();
  void executeInsideGlutLoop(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

 private:
  DevMemory devicePtr;
};


#endif /* GPU_KERNEL_H */
