#include <cuda_runtime.h>

#include "GpuKernel.h"
#include "Log.h"
#include "CudaHelpers.h"

GpuKernel::GpuKernel() {

}

GpuKernel::~GpuKernel() {

}

int GpuKernel::allocateDeviceMemory() {
  cout << "\t> Allocating device memory... ";
  
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.inputAtomsStructure, sizeof(Structure) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtomsStructure, sizeof(Structure) ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::sendDataToDevice(Structure * atomsStructure) {
  cout << "\t> Sending data to device... ";
  
  HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::executeKernel() {
  cout << "\t> Executing kernel... ";

  atomsStructureTest<<<1,1>>>( devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);

  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::getDataFromDevice() {
  Structure * tmpOutputData = new Structure();
  cout << "\t> Receiving data from device... ";

  HANDLE_ERROR( cudaMemcpy( tmpOutputData, devicePtr.outputAtomsStructure, sizeof(Structure), cudaMemcpyDeviceToHost ) );

  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::clearDeviceMemory() {
  HANDLE_ERROR( cudaFree( devicePtr.inputAtomsStructure ) );
  HANDLE_ERROR( cudaFree( devicePtr.outputAtomsStructure ) );

  cudaDeviceReset();
  return SUCCESS;
}

