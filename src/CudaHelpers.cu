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

__global__ void update_structure(Structure *input, Structure *output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  input->atomsCount = output->atomsCount;
  for (int i=tid ; i<input->atomsCount ; i+=blockDim.x) {
    input->atoms[i].pos.x = output->atoms[i].pos.x;// + time;
    input->atoms[i].pos.y = output->atoms[i].pos.y;
    input->atoms[i].pos.z = output->atoms[i].pos.z;
    input->atoms[i].force = output->atoms[i].force;
    input->atoms[i].acceleration = output->atoms[i].acceleration;
    input->atoms[i].status = output->atoms[i].status;
    input->atoms[i].fixed = output->atoms[i].fixed;
  }
}

__global__ void update_structure_and_display(float4 *pos, Structure *input, Structure *output) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float u, v, w;

  input->atomsCount = output->atomsCount;
  for (int i=tid ; i<input->atomsCount ; i+=blockDim.x) {
    input->atoms[i].pos.x = output->atoms[i].pos.x;// + time;
    input->atoms[i].pos.y = output->atoms[i].pos.y;
    input->atoms[i].pos.z = output->atoms[i].pos.z;
    input->atoms[i].force = output->atoms[i].force;
    input->atoms[i].acceleration = output->atoms[i].acceleration;
    input->atoms[i].status = output->atoms[i].status;
    input->atoms[i].fixed = output->atoms[i].fixed;

    u = input->atoms[i].pos.x * 0.1f;
    w = input->atoms[i].pos.y * 0.1f;
    v = input->atoms[i].pos.z * 0.1f;
    pos[i] = make_float4(u, w, v, 1.0f);
  }
}

__global__ void prepare_display(float4 *pos, Structure *input) {
  float u, v, w;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i=tid ; i<input->atomsCount ; i+=blockDim.x) {
    u = input->atoms[i].pos.x * 0.1f;
    w = input->atoms[i].pos.y * 0.1f;
    v = input->atoms[i].pos.z * 0.1f;
    pos[i] = make_float4(u, w, v, 1.0f);
  }
}

__global__ void MD_LJ_kernel(Structure *input, Structure *output, float time) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int atomIndexStart = tid;//blockIdx.x;
  int atomIndexEnd = input->atomsCount;
  float forceGradient[3] = {0.0f, 0.0f, 0.0f};
  
  // COMPUTING
  float dX = 0;
  float dY = 0;
  float dZ = 0;
  float distance = 0;
  float force = 0;
  float modifier = 1.0f;

 for (int i=atomIndexStart ; i<atomIndexEnd ; i += blockDim.x * gridDim.x) {
    forceGradient[0] = 0.0f;
    forceGradient[1] = 0.0f;
    forceGradient[2] = 0.0f;

    for (int j=0 ; j<input->atomsCount ; j++) {
      if (i == j)
	continue;
      
      force = 0;
      
      dX = input->atoms[j].pos.x - input->atoms[i].pos.x;
      dY = input->atoms[j].pos.y - input->atoms[i].pos.y;
      dZ = input->atoms[j].pos.z - input->atoms[i].pos.z;
      distance = sqrtf(pow(dX, 2) + pow(dY, 2) + pow(dZ, 2));
      
      if (distance <= 20|| distance >= 30)
	continue;
      
      // force 
      force = 4 * 1.0f/*E*/ * ( pow((3.0f/distance), 12) -  pow((3.0f/distance), 6) );

      forceGradient[0] += - (dX / distance) * force * input->atoms[i].force;
      forceGradient[1] += - (dY / distance) * force * input->atoms[i].force;
      forceGradient[2] += - (dZ / distance) * force * input->atoms[i].force;
    }

    output->atoms[i].pos.x = input->atoms[i].pos.x + forceGradient[0] * modifier;
    output->atoms[i].pos.y = input->atoms[i].pos.y + forceGradient[1] * modifier;
    output->atoms[i].pos.z = input->atoms[i].pos.z + forceGradient[2] * modifier;
 }
}

__global__ void vbo_MD_kernel(float4 *pos, Structure * input, float time)
{
  int atomsCount = input->atomsCount;
  int tmpCount = 0;
  float u, v, w;
  for (int i=0 ; (i<input->dim.x) && (tmpCount < atomsCount) ; i++) {
    for (int j=0 ; (j<input->dim.y) && (tmpCount < atomsCount) ; j++) {
      for (int k=0 ; (k<input->dim.z) && (tmpCount < atomsCount); k++) {
	u = input->atoms[tmpCount].pos.x * 0.1f;
	w = input->atoms[tmpCount].pos.y * 0.1f;
        v = input->atoms[tmpCount].pos.z * 0.1f;

	pos[tmpCount] = make_float4(u, w, v, 1.0f);
	tmpCount++;
      }
    }
  }
  /*
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int index = y * input->dim.x + x;

    float u = input->atoms[index].pos.x * 0.1f;

    float w = input->atoms[index].pos.y * 0.1f;

    float v = input->atoms[index].pos.z * 0.1f;
    

    pos[index] = make_float4(u, w, v, 1.0f);
  */
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

void handleTimerError(cudaError_t error, int type) {
  if (error == cudaSuccess)
    return;

  switch (type) {
  case START_CREATE:
    fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
    break;
  case STOP_CREATE:
    fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
    break;
  case START_RECORD:
    fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
    break;
  case STOP_RECORD:
    fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
    break;
  case SYNCHRONIZE:
    fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
    break;
  case ELAPSED_TIME:
    fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
    break;
  default:
    fprintf(stderr, "Unknown error!\n");
    break;
  }
  
  exit(EXIT_FAILURE);
}
