#include "Global.h"
#include "Structure.h"
#include "Simulation.h"

int main(int argc, char** argv) {
  Structure * atomsStruct = new Structure();
  atomsStruct->init();
  Simulation::instance()->perform();

  /*
  printf("[Molecular Dynamics Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -devLimit=devicesCount (Limit of devices used to computing - use 1 or numer of devices)\n");
    printf("      -size=size (Atoms Structure size)\n");
    printf("      -deviceList (Displays devices list)\n");

    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "deviceList")) {
    displayDevices();
    exit(EXIT_SUCCESS);
  }

  int cmdStructSize = 12;
  if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
    cmdStructSize = getCmdLineArgumentInt(argc, (const char **)argv, "size");
  }

  int deviceID = 0;
  int deviceCount = 1;
  getDevices(argc, argv, deviceID, deviceCount);
  displayDevices();


  AtomsStructure *hostStructure = new AtomsStructure(cmdStructSize);
  AtomsStructure *deviceData = new AtomsStructure[deviceCount];
    
  float *hostResult = new float[hostStructure->Size()];

  prepareDeviceInputData(hostStructure, deviceData, deviceCount);

  // Prepared data test
  // for (int i=0 ; i<deviceCount ; i++) {
  //     for (int j=0 ; j<hostStructure->Size() / deviceCount ; j++)
  //     printf("%d) X=%f Y=%f Z=%f\n", i, deviceData[i].x[j], deviceData[i].y[j], deviceData[i].z[j]);
  // }

  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal = 0.0f;


  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  pthread_t * threads = new pthread_t[deviceCount];
  for (int i=0 ; i<deviceCount ; i++)
    threads[i] = startThread(executeKernel, &(deviceData[i]));
  for (int i=0 ; i<deviceCount ; i++)
    endThread( threads[i] );

  HANDLE_ERROR(cudaEventRecord(stop, NULL));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

  float msecPerMatrixMul = msecTotal / 1;
  double flopsPerMatrixMul = 2.0 ;//* (128*((double)hostStructure->Size() + (double)hostStructure->Size() + (double)hostStructure->Size()));//(double)hostStructure->Size() * (double)hostStructure->Size() * (double)hostStructure->Size();
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

  fprintf( stderr,
	   "Performance= %.2f GFlop/s, Time= %.10f msec",
	   gigaFlops,
	   msecPerMatrixMul);

  mergeResult(hostResult, deviceData, deviceCount);

  // Test result
  bool correct = true;
  float correctResult = 1.f ;//* 3.f;
  for (int i=0 ; i< hostStructure->Size() ; i++)
    if (hostResult[i] != correctResult)
      correct = false;
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  // ~!
  */
  return 0;
}

