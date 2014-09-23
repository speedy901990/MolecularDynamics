#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

#include <stdio.h>
#include "helper_functions.h"
#include "Structure.h"

//------------------- Kernels ----------------------------------
__global__ void add( float *x, float *y, float *z, float *result , int size);
__global__ void multiply( float *x, float *y, float *z, float *result , int size);
__global__ void atomsStructureTest( Structure * input, Structure * output);
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
__global__ void vbo_MD_kernel(float4 *pos, Structure * input, float time);
__global__ void update_structure(Structure *input, Structure *output);
__global__ void update_structure_and_display(float4 *pos, Structure *input, Structure *output);
__global__ void MD_LJ_kernel( Structure *input, Structure *output, float time = 0);

// ERROR handling-----------------------------------------------------------

void HandleError( cudaError_t err, const char *file, int line );
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

// Threading for multi GPU support------------------------------------------
typedef void *(*CUT_THREADROUTINE)(void *);
pthread_t startThread(CUT_THREADROUTINE func, void * data);
void endThread(pthread_t thread);

// CUDA Timer event error handler
enum {
  START_CREATE,
  STOP_CREATE,
  START_RECORD,
  STOP_RECORD,
  SYNCHRONIZE,
  ELAPSED_TIME
};
void handleTimerError(cudaError_t error, int type);

// Other helper methodes-----------------------------------------------------
void displayAvailableDevices();
void displayChosenDevices(int * devicesID, int devicesCount);
void getDevices(int * &devicesID, int &devicesCount);


/*
void prepareDeviceInputData(AtomsStructure *hostStructure, AtomsStructure *deviceData, int deviceCount) {
    for (int i=0 ; i<deviceCount ; i++) {
        deviceData[i] = AtomsStructure(hostStructure->Size() / deviceCount);
        deviceData[i].x = hostStructure->x + hostStructure->Size() / deviceCount * i;
        deviceData[i].y = hostStructure->y + hostStructure->Size() / deviceCount * i;
        deviceData[i].z = hostStructure->z + hostStructure->Size() / deviceCount * i;
        deviceData[i].deviceID = i;
        deviceData[i].result = new float[hostStructure->Size() / deviceCount];
        deviceData[i].iterN = hostStructure->iterN / deviceCount;
    }
}

void mergeResult(float * hostResult, AtomsStructure *deviceData, int deviceCount) {
    int counter = 0;
    for (int i=0 ; i<deviceCount; i++)
        for (int j=0 ; j<deviceData[i].Size(); j++) {
            hostResult[counter] = deviceData[i].result[j];
            //printf("%d) %f\t", counter, deviceData[i].result[j]);
            counter++;
        }
    printf("\n");
}

void * executeKernel(void * threadData) {
    setbuf(stdout, NULL);
    AtomsStructure * data = (AtomsStructure *)threadData;
    cudaSetDevice(data->deviceID);
    int size = data->Size();
    float * deviceX, * deviceY, * deviceZ, * deviceResult;
    float * newX, * newY, * newZ;

    HANDLE_ERROR( cudaMalloc( (void**)&deviceX, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceY, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceZ, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceResult, size * sizeof(float) ) );
    // HANDLE_ERROR( cudaMalloc( (void**)&newX, size * sizeof(float) ) );
    // HANDLE_ERROR( cudaMalloc( (void**)&newY, size * sizeof(float) ) );
    // HANDLE_ERROR( cudaMalloc( (void**)&newZ, size * sizeof(float) ) );

    HANDLE_ERROR( cudaMemcpy( deviceX, data->x, size * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( deviceY, data->y, size * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( deviceZ, data->z, size * sizeof(float), cudaMemcpyHostToDevice ) );
    
    fprintf(stderr, "Computing result using CUDA Kernel...\n");
    //add<<<(size + 127 ) /128, 128>>>( deviceX,deviceY, deviceZ, deviceResult, size);
    for (int i=0 ; i<data->iterN ; i++)
        multiply<<<(size + 127 ) /128, 128>>>( deviceX,deviceY, deviceZ, deviceResult, size);
    //lennardSolver<<<(size + 127 ) /128, 512>>>( deviceX,deviceY, deviceZ, newX, newY, newZ, size);
    fprintf(stderr, "done\n");

    HANDLE_ERROR( cudaMemcpy( data->result, deviceResult, size * sizeof(float), cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaFree( deviceX ) );
    HANDLE_ERROR( cudaFree( deviceY ) );
    HANDLE_ERROR( cudaFree( deviceZ ) );
    HANDLE_ERROR( cudaFree( deviceResult ) );
    // HANDLE_ERROR( cudaFree( newX ) );
    // HANDLE_ERROR( cudaFree( newY ) );
    // HANDLE_ERROR( cudaFree( newZ ) );

    //printf("%s\n", data->result ? "Partial Result = NOT NULL" : "Result = NULL");
}
*/
#endif  // CUDAHELPERS_H
