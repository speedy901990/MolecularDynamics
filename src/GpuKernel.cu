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
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtomsStructure, sizeof(Structure) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtoms, sizeof(Atom) * atomsStructure->atomsCount ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::sendDataToDevice(Structure * &atomsStructure) {
  cout << "\t> Sending data to device... " << flush;
  
  HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtoms, atomsStructure->atoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( &(devicePtr.inputAtomsStructure->atoms), &(devicePtr.inputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );

  HANDLE_ERROR( cudaMemcpy( devicePtr.outputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( devicePtr.outputAtoms, atomsStructure->atoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( &(devicePtr.outputAtomsStructure->atoms), &(devicePtr.outputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::execute(Structure * structure, bool displayOn) {
  cout << flush << "\t> Executing kernel... "<< flush;
  if (structure == NULL) {
    Log::instance()->toConsole(E_NULL_PTR, typeid(this).name(), __FUNCTION__, __LINE__, "Structure is NULL.");
    exit(EXIT_FAILURE);
  }
  
  this->structure = structure;

  if (displayOn) {
    executeDisplayOn();
  }
  else {
    executeDisplayOff();
    cout << "\t...done!" << endl << flush;
  }

  return SUCCESS;
}

int GpuKernel::executeDisplayOn() {
  GpuDisplay::instance()->runAnimation(this);

  return SUCCESS;
}

int GpuKernel::executeDisplayOff() {
  int mesh_width = structure->dim.x;
  int mesh_height = structure->dim.y;
  int threadsPerBlock = 256;
  int blocksPerGrid = (mesh_width * mesh_width * mesh_width + threadsPerBlock - 1) / threadsPerBlock;
  dim3 block(threadsPerBlock, 1, 1);
  dim3 grid(blocksPerGrid, 1, 1);
  int nIter = 1;
  cudaError_t error;
  float msecTotal = 0.0f;

  cudaEvent_t start;
  handleTimerError(cudaEventCreate(&start), START_CREATE);

  cudaEvent_t stop;
  handleTimerError(cudaEventCreate(&stop), STOP_CREATE);

  update_structure<<< 1, 1 >>>(devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);
  cudaDeviceSynchronize();

  handleTimerError(cudaEventRecord(start, NULL), START_RECORD);

  MD_LJ_kernel<<< grid, block >>>(devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);
  cudaDeviceSynchronize();

  handleTimerError(cudaEventRecord(stop, NULL), STOP_RECORD);
  handleTimerError(cudaEventSynchronize(stop), SYNCHRONIZE);
  handleTimerError(cudaEventElapsedTime(&msecTotal, start, stop), ELAPSED_TIME);
  displayPerformanceResults(msecTotal, nIter, block, grid);
  
  return SUCCESS;
}

void GpuKernel::displayPerformanceResults(float msecTotal, int nIter, dim3 block, dim3 grid) {
  float msecPerMatrixMul = msecTotal / nIter;
  //double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
  double flopsPerMatrixMul = 27.0 * (double)structure->dim.x * (double)structure->dim.y * (double)structure->dim.z
                           + 6.0 * (double)structure->dim.x * (double)structure->dim.y;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("\n\t\tPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
	 gigaFlops,
	 msecPerMatrixMul,
	 flopsPerMatrixMul,
	 block.x * block.y);
}

void GpuKernel::executeInsideGlutLoop(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (mesh_width * mesh_width * mesh_width + threadsPerBlock - 1) / threadsPerBlock;
  dim3 block(threadsPerBlock, 1, 1);
  dim3 grid(blocksPerGrid, 1, 1);

  update_structure<<< 1, 1 >>>(devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);
  cudaDeviceSynchronize();
  MD_LJ_kernel<<< grid, block >>>(devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure, time);
  cudaDeviceSynchronize();
  prepare_display<<< 1, 1 >>>(pos, devicePtr.inputAtomsStructure); 
  cudaDeviceSynchronize();
}

int GpuKernel::getDataFromDevice(Structure *&atomsStructure) {
  Structure * tmpOutputData = new Structure();
  Atom * atoms = new Atom[atomsStructure->atomsCount];

  cout << "\t> Receiving data from device... ";
  
  HANDLE_ERROR( cudaMemcpy( tmpOutputData, devicePtr.outputAtomsStructure, sizeof(Structure), cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( /*tmpOutputData->atoms*/atoms, /*devicePtr.outputAtomsStructure->atoms*/devicePtr.outputAtoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyDeviceToHost ) );

  cout << "done!" << endl;
  /*
  cout << "Data:" << endl;
  for (int i=0 ; i<atomsStructure->atomsCount ; i++) {
    cout << "Atom " << i << " x=" << atoms[i].pos.x << " y=" << atoms[i].pos.y << " z=" << atoms[i].pos.z
	 <<endl;//	 << " gradientX=" << atoms[i].gradientX << " gradientY=" << atoms[i].gradientY << " gradientZ=" << atoms[i].gradientZ << " force=" << atoms[i].force <<  endl;
  }
  */
  return SUCCESS;
}

int GpuKernel::clearDeviceMemory() {
  HANDLE_ERROR( cudaFree( devicePtr.inputAtomsStructure ) );
  HANDLE_ERROR( cudaFree( devicePtr.outputAtomsStructure ) );
  HANDLE_ERROR( cudaFree( devicePtr.inputAtoms ) );
  HANDLE_ERROR( cudaFree( devicePtr.outputAtoms ) );

  cudaDeviceReset();

  return SUCCESS;
}

