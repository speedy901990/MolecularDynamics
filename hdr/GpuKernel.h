#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include "Global.h"
#include "Structure.h"
#include "GpuDisplay.h"
#include "PerformanceStatistics.h"

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
  int allocateDeviceMemory(Structure * &atomsStructure, int deviceCount);
  int sendDataToDevice(Structure * &atomsStructure, int devicesCount);
  int execute(Structure * structure, int devicesCount, bool displayOn = true);
  int getDataFromDevice(Structure *&atomsStructure, int devicesCount);
  int clearDeviceMemory(int devicesCount);
  int executeVisualOn();
  int executeVisualOff(int devicesCount);
  PerformanceStatistics * executeThreadKernel(int tid);
  void executeInsideGlutLoop(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

 private:
  DevMemory * devicePtr;
  Structure * structure;

  void displayPerformanceResults(PerformanceStatistics *p);
};


#endif /* GPU_KERNEL_H */
