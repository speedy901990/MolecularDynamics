#ifndef __CUDAHELPERS_H__
#define __CUDAHELPERS_H__
#include <stdio.h>
#include "helper_functions.h"

//------------------- Kernels ----------------------------
__global__ void add( float *x, float *y, float *z, float *result , int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Ten w¹tek przetwarza dane pod okreœlonym indeksem
    while (tid < size) {
        result[tid] = x[tid] + y[tid] + z[tid];
        tid += blockDim.x * gridDim.x;
    }
}
//--------------------------------------------------------

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

// Threading for more GPU support
typedef void *(*CUT_THREADROUTINE)(void *);

pthread_t startThread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

void endThread(pthread_t thread){
    pthread_join(thread, NULL);
}

void displayDevices() {
    int deviceCount;
    cudaError_t error;
    cudaDeviceProp deviceProp;

    error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
        printf("cudaGetDeviceCount returned error code %d, line(%d)\n", error, __LINE__);

    printf("Available devices: %d\n", deviceCount);

    for (int i=0 ; i<deviceCount ; i++) {
        error = cudaGetDeviceProperties(&deviceProp, i);
        if (error != cudaSuccess)
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    printf("\n");
}

void getDevices(int argc, char** argv, int &deviceID, int &deviceCount) {
    cudaError_t error;

    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        deviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(deviceID);;

        if (error != cudaSuccess) {
            printf("cudaSetDevice returned error code, %d, line(%d) - no such device\n", error, __LINE__);
            exit(EXIT_SUCCESS);
        } 
    }  
    else {
        error = cudaGetDeviceCount(&deviceCount);

        if (error != cudaSuccess)
            printf("cudaGetDeviceCount returned error code %d, line(%d)\n", error, __LINE__);
        if (checkCmdLineFlag(argc, (const char **)argv, "devLimit")) {
            deviceCount = getCmdLineArgumentInt(argc, (const char **)argv, "devLimit");
            if (deviceCount%2 != 0 && deviceCount != 1) {
                printf("use 1 or even number of devices limit\n");
                exit(EXIT_SUCCESS);
            }
        }
    }
}

void prepareDeviceInputData(AtomsStructure *hostStructure, AtomsStructure *deviceData, int deviceCount) {
    for (int i=0 ; i<deviceCount ; i++) {
        deviceData[i] = AtomsStructure(hostStructure->Size() / deviceCount);
        deviceData[i].x = hostStructure->x + hostStructure->Size() / deviceCount * i;
        deviceData[i].y = hostStructure->y + hostStructure->Size() / deviceCount * i;
        deviceData[i].z = hostStructure->z + hostStructure->Size() / deviceCount * i;
        deviceData[i].deviceID = i;
        deviceData[i].result = new float[hostStructure->Size() / deviceCount];
    }
}

void mergeResult(float * hostResult, AtomsStructure *deviceData, int deviceCount) {
    int counter = 0;
    for (int i=0 ; i<deviceCount; i++)
        for (int j=0 ; j<deviceData[i].Size(); j++) {
            hostResult[counter] = deviceData[i].result[j];
            printf("%d) %f\t", counter, deviceData[i].result[j]);
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

    HANDLE_ERROR( cudaMalloc( (void**)&deviceX, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceY, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceZ, size * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&deviceResult, size * sizeof(float) ) );

    HANDLE_ERROR( cudaMemcpy( deviceX, data->x, size * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( deviceY, data->y, size * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( deviceZ, data->z, size * sizeof(float), cudaMemcpyHostToDevice ) );
    
    fprintf(stderr, "Computing result using CUDA Kernel...\n");
    add<<<128, 128>>>( deviceX,deviceY, deviceZ, deviceResult, size);
    fprintf(stderr, "done\n");

    HANDLE_ERROR( cudaMemcpy( data->result, deviceResult, size * sizeof(float), cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaFree( deviceX ) );
    HANDLE_ERROR( cudaFree( deviceY ) );
    HANDLE_ERROR( cudaFree( deviceZ ) );
    HANDLE_ERROR( cudaFree( deviceResult ) );

    printf("%s\n", data->result ? "Partial Result = NOT NULL" : "Result = NULL");
}

#endif  // __CUDAHELPERS_H__
