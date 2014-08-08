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

__global__ void multiply( float *x, float *y, float *z, float *result , int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;    // Ten w¹tek przetwarza dane pod okreœlonym indeksem
    while (tid < size) {
        result[tid] = x[tid] * y[tid] * z[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void lennardSolver( float * X,
                             float * Y,
                             float * Z,
                             float * newX,
                             float * newY,
                             float * newZ,
                             int atomsCount)
{
    // get global id
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    // get global size
    int globalSize = gridDim.x;

    int numberOfElementsForOne = (atomsCount) / globalSize;
    // nadmiar
    int excess = (atomsCount) - (numberOfElementsForOne * globalSize);


    int offset, end;
    if(globalId < excess)
    {
        offset = globalId * (numberOfElementsForOne + 1);
        end = offset + numberOfElementsForOne + 1;
    }
    else
    {
        offset = globalId * numberOfElementsForOne + excess;
        end = offset + numberOfElementsForOne;
    }


    // main algorithm

    //float delta = 0.001;

    // potential coefficients
    //float e = 1.5;
    //float a = 1.0;

    // zeros gradient
    float forceGradient[3] = {0.0f, 0.0f, 0.0f};
    
    for(int atomIndex = offset; atomIndex < end; ++atomIndex)
    {

    forceGradient[0] = 0.0f;
    forceGradient[1] = 0.0f;
    forceGradient[2] = 0.0f;
            
        for(int i = 0; i < atomsCount; ++i)
        {
            // jesli indeksy nie wskazuja na ten sam atom
            if(i != atomIndex)
            {
                float distanceX = X[i] - X[atomIndex];
                float distanceY = Y[i] - Y[atomIndex];
                float distanceZ = Z[i] - Z[atomIndex];

                // calculate distance beetwen atoms
                float distance = //sqrt(
                                distanceX * distanceX
                               + distanceY * distanceY
                               + distanceZ * distanceZ
                               //)
                               ;

                // cut distance
                if(distance <= 2.5f * 2.5f)
                {
                    float force = 24 * 1.5f * (
                                //2 * pow((1.0/distance), 13) -
                                2 * pow((1.0f/distance), 6.5f) -
                                //pow((1.0/distance), 7)
                                pow((1.0f/distance), 3.5f)
                            );// / a;

                    // force gradient
                    forceGradient[0] += - (distanceX / distance) * force;
                    forceGradient[1] += - (distanceY / distance) * force;
                    forceGradient[2] += - (distanceZ / distance) * force;
                }
            }
        }

        // calculate new position
        newX[atomIndex] = X[atomIndex]
                        + forceGradient[0] * 0.001f;

        newY[atomIndex] = Y[atomIndex]
                        + forceGradient[1] * 0.001f;

        newZ[atomIndex] = Z[atomIndex]
                        + forceGradient[2] * 0.001f;
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

// Threading for multi GPU support
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
#endif  // __CUDAHELPERS_H__
