#include <cuda_runtime.h>

#include "GpuKernel.h"
#include "Log.h"
#include "CudaHelpers.h"
#include "GpuThread.h"

GpuKernel::GpuKernel() {

}

GpuKernel::~GpuKernel() {

}

int GpuKernel::allocateDeviceMemory(Structure * &atomsStructure, int deviceCount) {
  cout << "\t> Allocating device memory... ";
  
  if (false/*deviceCount == 1*/) {
    /*HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.inputAtomsStructure, sizeof(Structure) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.inputAtoms, sizeof(Atom) * atomsStructure->atomsCount ) );
    HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtomsStructure, sizeof(Structure) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&devicePtr.outputAtoms, sizeof(Atom) * atomsStructure->atomsCount ) );*/
  }
  else {
    devicePtr = new DevMemory[deviceCount];
    for (int i=0 ; i<deviceCount ; i++) {
      cudaSetDevice(i);
      HANDLE_ERROR( cudaMalloc( (void**)&devicePtr[i].inputAtomsStructure, sizeof(Structure) ) );
      HANDLE_ERROR( cudaMalloc( (void**)&devicePtr[i].inputAtoms, sizeof(Atom) * atomsStructure[i].atomsCount ) );
      HANDLE_ERROR( cudaMalloc( (void**)&devicePtr[i].outputAtomsStructure, sizeof(Structure) ) );
      HANDLE_ERROR( cudaMalloc( (void**)&devicePtr[i].outputAtoms, sizeof(Atom) * atomsStructure[i].atomsCount ) );
    }
  }

  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::sendDataToDevice(Structure * &atomsStructure, int deviceCount) {
  cout << "\t> Sending data to device... " << flush;
  
  if (false/*deviceCount == 1*/) {
    /*HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( devicePtr.inputAtoms, atomsStructure->atoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(devicePtr.inputAtomsStructure->atoms), &(devicePtr.inputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( devicePtr.outputAtomsStructure, atomsStructure, sizeof(Structure), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( devicePtr.outputAtoms, atomsStructure->atoms, sizeof(Atom) * atomsStructure->atomsCount, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(devicePtr.outputAtomsStructure->atoms), &(devicePtr.outputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );*/
  }
  else {
    for (int i=0 ; i<deviceCount ; i++) {
      cudaSetDevice(i);
      HANDLE_ERROR( cudaMemcpy( devicePtr[i].inputAtomsStructure, atomsStructure + i, sizeof(Structure), cudaMemcpyHostToDevice ) );
      HANDLE_ERROR( cudaMemcpy( devicePtr[i].inputAtoms, atomsStructure[i].atoms, sizeof(Atom) * atomsStructure[i].atomsCount, cudaMemcpyHostToDevice ) );
      HANDLE_ERROR( cudaMemcpy( &(devicePtr[i].inputAtomsStructure->atoms), &(devicePtr[i].inputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );

      HANDLE_ERROR( cudaMemcpy( devicePtr[i].outputAtomsStructure, atomsStructure + i, sizeof(Structure), cudaMemcpyHostToDevice ) );
      HANDLE_ERROR( cudaMemcpy( devicePtr[i].outputAtoms, atomsStructure[i].atoms, sizeof(Atom) * atomsStructure[i].atomsCount, cudaMemcpyHostToDevice ) );
      HANDLE_ERROR( cudaMemcpy( &(devicePtr[i].outputAtomsStructure->atoms), &(devicePtr[i].outputAtoms), sizeof(Atom *), cudaMemcpyHostToDevice ) );
    }
  }
  
  cout << "done!" << endl;

  return SUCCESS;
}

int GpuKernel::execute(Structure * structure, int devicesCount, bool displayOn) {
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
    if (devicesCount != 1)
      executeMultiGpu(devicesCount);
    else
      executeDisplayOff();
    cout << "\t...done!" << endl << flush;
  }

  return SUCCESS;
}

int GpuKernel::executeDisplayOn() {
  GpuDisplay::instance()->runAnimation(this);

  return SUCCESS;
}

int GpuKernel::executeMultiGpu(int deviceCount) {
  
  pthread_t * threads = new pthread_t[deviceCount];
  GpuThread * threadsData = new GpuThread[deviceCount];

  for (int i=0 ; i<deviceCount ; i++) {
    threadsData[i].kernel = this;
    threadsData[i].tid = i;
    threads[i] = startThread(executeGpuThreadKernel, (void *)threadsData);
  }
  
  for (int i=0 ; i<deviceCount ; i++)
    endThread( threads[i] );
  
  printf("------- TOTAL PERFORMANCE: --------");
  //TODO FIX segfault
  //for (int i=0 ; i<deviceCount ; i++)
  //  displayPerformanceResults(threadsData[i].performance);
  
  return SUCCESS;
}

PerformanceStatistics * GpuKernel::executeThreadKernel(int tid) {
  int mesh_width = structure[tid].dim.x;
  int mesh_height = structure[tid].dim.y;
  int threadsPerBlock = 1024;
  int blocksPerGrid = (mesh_width * mesh_width * mesh_width + threadsPerBlock - 1) / threadsPerBlock;
  dim3 block(threadsPerBlock, 1, 1);
  dim3 grid(blocksPerGrid, 1, 1);
  int nIter = 100;
  cudaError_t error;
  float msecTotal = 0.0f;
  
  cudaEvent_t start;
  handleTimerError(cudaEventCreate(&start), START_CREATE);
  
  cudaEvent_t stop;
  handleTimerError(cudaEventCreate(&stop), STOP_CREATE);
  
  handleTimerError(cudaEventRecord(start, NULL), START_RECORD);
  /*
  for (int i=0 ; i<nIter ; i++) {
    update_structure<<< grid, block >>>(devicePtr[tid].inputAtomsStructure, devicePtr[tid].outputAtomsStructure);
    MD_LJ_kernel<<< grid, block >>>(devicePtr[tid].inputAtomsStructure, devicePtr[i].outputAtomsStructure);
  }
  
  cudaDeviceSynchronize();
  */
  handleTimerError(cudaEventRecord(stop, NULL), STOP_RECORD);
  handleTimerError(cudaEventSynchronize(stop), SYNCHRONIZE);
  handleTimerError(cudaEventElapsedTime(&msecTotal, start, stop), ELAPSED_TIME);
  
  PerformanceStatistics * performance = new PerformanceStatistics(msecTotal, nIter, block, grid);
  displayPerformanceResults(performance);
  
  return performance;
}

int GpuKernel::executeDisplayOff() {
  int mesh_width = structure->dim.x;
  int mesh_height = structure->dim.y;
  int threadsPerBlock = 1024;
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
  
  handleTimerError(cudaEventRecord(start, NULL), START_RECORD);

  for (int i=0 ; i<nIter ; i++) {
    update_structure<<< grid, block >>>(devicePtr[i].inputAtomsStructure, devicePtr[i].outputAtomsStructure);
    MD_LJ_kernel<<< grid, block >>>(devicePtr[i].inputAtomsStructure, devicePtr[i].outputAtomsStructure);
  }

  cudaDeviceSynchronize();

  handleTimerError(cudaEventRecord(stop, NULL), STOP_RECORD);
  handleTimerError(cudaEventSynchronize(stop), SYNCHRONIZE);
  handleTimerError(cudaEventElapsedTime(&msecTotal, start, stop), ELAPSED_TIME);

  PerformanceStatistics * performance = new PerformanceStatistics(msecTotal, nIter, block, grid);
  displayPerformanceResults(performance);
  
  return SUCCESS;
}

void GpuKernel::displayPerformanceResults(PerformanceStatistics *p) {
  float msecPerSimulation = p->msecTotal / p->nIter;
  double flopsPerSimulation = 55.0 * structure->atomsCount * structure->atomsCount + 10.0 * structure->atomsCount + 2 * (structure->atomsCount + 256 - 1 )/ 256;
  double gigaFlops = (flopsPerSimulation * 1.0e-9f) / (msecPerSimulation / 1000.0f);
  printf("\n\t\tPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
	 gigaFlops,
	 msecPerSimulation,
	 flopsPerSimulation,
	 p->block.x * p->block.y);
}

void GpuKernel::executeInsideGlutLoop(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (mesh_width * mesh_width * mesh_width + threadsPerBlock - 1) / threadsPerBlock;
  dim3 block(threadsPerBlock, 1, 1);
  dim3 grid(blocksPerGrid, 1, 1);

  //  update_structure_and_display<<< grid, block >>>(pos, devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure);
  update_structure_and_display<<< grid, block >>>(pos, devicePtr[0].inputAtomsStructure, devicePtr[0].outputAtomsStructure);
  //  MD_LJ_kernel<<< grid, block >>>(devicePtr.inputAtomsStructure, devicePtr.outputAtomsStructure, time);
  MD_LJ_kernel<<< grid, block >>>(devicePtr[0].inputAtomsStructure, devicePtr[0].outputAtomsStructure, time);
  cudaDeviceSynchronize();
}

int GpuKernel::getDataFromDevice(Structure *&atomsStructure, int deviceCount) {
  Structure * tmpOutputData = new Structure();
  Atom * atoms = new Atom[atomsStructure->atomsCount];

  cout << "\t> Receiving data from device... ";

  for (int i=0 ; i<1 ; i++) {
    HANDLE_ERROR( cudaMemcpy( tmpOutputData, devicePtr[i].outputAtomsStructure, sizeof(Structure), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( /*tmpOutputData->atoms*/atoms, /*devicePtr.outputAtomsStructure->atoms*/devicePtr[i].outputAtoms, sizeof(Atom) * atomsStructure[i].atomsCount, cudaMemcpyDeviceToHost ) );
  }
  cout << "done!" << endl;
  
  /*cout << "Data:" << endl;
  for (int i=0 ; i<atomsStructure->atomsCount ; i++) {
    cout << "Atom " << i << " x=" << atoms[i].pos.x << " y=" << atoms[i].pos.y << " z=" << atoms[i].pos.z
	 <<endl;//	 << " gradientX=" << atoms[i].gradientX << " gradientY=" << atoms[i].gradientY << " gradientZ=" << atoms[i].gradientZ << " force=" << atoms[i].force <<  endl;
  }
  */
  return SUCCESS;
}

int GpuKernel::clearDeviceMemory(int devicesCount) {
  if (false/*devicesCount == 1*/) {
    /*HANDLE_ERROR( cudaFree( devicePtr.inputAtomsStructure ) );
    HANDLE_ERROR( cudaFree( devicePtr.outputAtomsStructure ) );
    HANDLE_ERROR( cudaFree( devicePtr.inputAtoms ) );
    HANDLE_ERROR( cudaFree( devicePtr.outputAtoms ) );*/
  }
  else {
    for (int i=0 ; i<devicesCount ; i++) {
      HANDLE_ERROR( cudaFree( devicePtr[i].inputAtomsStructure ) );
      HANDLE_ERROR( cudaFree( devicePtr[i].outputAtomsStructure ) );
      HANDLE_ERROR( cudaFree( devicePtr[i].inputAtoms ) );
      HANDLE_ERROR( cudaFree( devicePtr[i].outputAtoms ) );
    }
  }

  cudaDeviceReset();

  return SUCCESS;
}
