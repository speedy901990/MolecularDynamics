#include <cuda_runtime.h>

#include "GpuKernel.h"
#include "Log.h"
#include "CudaHelpers.h"

GpuKernel::GpuKernel() {

}

GpuKernel::~GpuKernel() {

}

int GpuKernel::allocateDeviceMemory(Structure * &atomsStructure) {
  cout << "\t> Allocating device memory... ";
  
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.inputAtomsStructure, sizeof(Structure) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.inputAtoms, sizeof(Atom) * atomsStructure->atomsCount ) );
  //HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtomsStructure, sizeof(Structure) ) );
  //HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtoms, sizeof(Atom) * atomsStructure->atomsCount ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::sendDataToDevice(Structure * &atomsStructure) {
  cout << "\t> Sending data to device... " << flush;
  
  HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtoms, atomsStructure->atoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( &(devicePtr.inputAtomsStructure->atoms), &(devicePtr.inputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::execute(bool displayOn) {
  cout << flush << "\t> Executing kernel... "<< flush;

  if (displayOn) {
    executeDisplayOn();
  }
  else {
    executeDisplayOff();
  }
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::executeDisplayOn() {
  GpuDisplay::instance()->runAnimation(this);

  return SUCCESS;
}

int GpuKernel::executeDisplayOff() {
  atomsStructureTest<<<1,1>>>( devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);

  return SUCCESS;
}

void GpuKernel::executeInsideGlutLoop(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time) {
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  //vbo_MD_kernel<<<grid, block>>>(pos, devicePtr.inputAtomsStructure, time);
  vbo_MD_kernel<<<1,1>>>(pos, devicePtr.inputAtomsStructure, time);
}

int GpuKernel::getDataFromDevice() {
  Structure * tmpOutputData = new Structure();
  cout << "\t> Receiving data from device... ";
  // TODO@@@@@@@@@@
  //  HANDLE_ERROR( cudaMemcpy( tmpOutputData, devicePtr.outputAtomsStructure, sizeof(Structure), cudaMemcpyDeviceToHost ) );

  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::clearDeviceMemory() {
  HANDLE_ERROR( cudaFree( devicePtr.inputAtomsStructure ) );
  HANDLE_ERROR( cudaFree( devicePtr.outputAtomsStructure ) );

  cudaDeviceReset();
  return SUCCESS;
}

