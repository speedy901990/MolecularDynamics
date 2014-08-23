#include "CudaHelpers.h"

//------------------- Kernels ----------------------------------
__global__ void add( float *x, float *y, float *z, float *result , int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        result[tid] = x[tid] + y[tid] + z[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void multiply( float *x, float *y, float *z, float *result , int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        result[tid] = x[tid] * y[tid] * z[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void atomsStructureTest( Structure * input, Structure * output) {
  //    int tid = threadIdx.x + blockIdx.x * blockDim.x;
  output->atomsCount = input->atomsCount;
}

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

__global__ void vbo_MD_kernel(float4 *pos, Structure * input, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int index = y * input->dim.x + x;
    // calculate uv coordinates
    /*float u = x / (float) input->dim.x;
    float v = y / (float) input->dim.y;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
    */
    float u = input->atoms[index].pos.x * 0.1f;
    //float u = 0.5f;//input->atoms[0].pos.x * 0.5f;
    float w = input->atoms[index].pos.y * 0.1f;
    float v = 0.5f*sinf(u*4.0f + time) * cosf(v*4.0f + time) * 0.5f;
    //float v = input->atoms[index].pos.z * 0.1f;
    
    //pos[index] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    pos[index] = make_float4(u, w, v, 1.0f);
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

// ERROR handling-----------------------------------------------------------

void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// Threading for multi GPU support------------------------------------------
typedef void *(*CUT_THREADROUTINE)(void *);

pthread_t startThread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

void endThread(pthread_t thread){
    pthread_join(thread, NULL);
}

// Other helper methodes-----------------------------------------------------
void displayAvailableDevices() {
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
}

void displayChosenDevices(int * devicesID, int devicesCount) {
  cudaError_t error;
  cudaDeviceProp deviceProp;
  printf("Chosen devices: %d\n", devicesCount);

  for (int i=0 ; i<devicesCount ; i++) {
        error = cudaGetDeviceProperties(&deviceProp, devicesID[i]);
        if (error != cudaSuccess)
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devicesID[i], deviceProp.name, deviceProp.major, deviceProp.minor);
    }
}

void getDevices(int * &devicesID, int &devicesCount) {
  cudaError_t error;
  int devicesLimit;
  error = cudaGetDeviceCount(&devicesLimit);
  if (error != cudaSuccess) {
    printf("cudaGetDeviceCount returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (devicesCount > devicesLimit) {
    printf("ERR: devicesCount cannot be larger than devicesLimit returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

      // TODO
  for (int i=0 ; i<devicesCount ; i++) {
    error = cudaSetDevice(devicesID[i]);;
    if (error != cudaSuccess) {
      printf("cudaSetDevice returned error code, %d, line(%d) - no such device\n", error, __LINE__);
      exit(EXIT_SUCCESS);
    }
  } 
}
