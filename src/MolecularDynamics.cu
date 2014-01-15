#include "Global.h"

void prepareInputStructure(AtomsStructure * provData, int deviceCount, int block_size, dim3 dimsA, dim3 dimsB)
{
    for (int i=0 ; i<deviceCount ; i++)
    {
        // provData[i].deviceID = i;
        // provData[i].deviceCount = deviceCount;
        // provData[i].blockSize = block_size;
        // provData[i].dimsA.x = dimsA.x / deviceCount;
        // provData[i].dimsA.y = dimsA.y / deviceCount;
        // provData[i].dimsB.x = dimsB.x / deviceCount;
        // provData[i].dimsB.x = dimsB.x / deviceCount;
    }
    //result = new Result[deviceCount];
}

int main(int argc, char** argv) {
    printf("[Molecular Dynamics Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -devLimit=devicesCount (Limit of devices used to computing - use 1 or even numer of devices)\n");
        printf("      -size=size (Atoms Structure size)\n");
        printf("      -deviceList (Displays devices list)\n");

        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "deviceList")) {
        displayDevices();
        exit(EXIT_SUCCESS);
    }

    int deviceID = 0;
    int deviceCount = 1;
    getDevices(argc, argv, deviceID, deviceCount);
    displayDevices();


    AtomsStructure *hostStructure = new AtomsStructure(12);
    AtomsStructure *deviceData = new AtomsStructure[deviceCount];
    
    float *hostResult = new float[hostStructure->Size()];

    prepareDeviceInputData(hostStructure, deviceData, deviceCount);

    // Prepared data test
    // for (int i=0 ; i<deviceCount ; i++) {
    //     for (int j=0 ; j<hostStructure->Size() / deviceCount ; j++)
    //     printf("%d) X=%f Y=%f Z=%f\n", i, deviceData[i].x[j], deviceData[i].y[j], deviceData[i].z[j]);
    // }

    pthread_t * threads = new pthread_t[deviceCount];
    for (int i=0 ; i<deviceCount ; i++)
        threads[i] = startThread(executeKernel, &(deviceData[i]));
    for (int i=0 ; i<deviceCount ; i++)
        endThread( threads[i] );

    mergeResult(hostResult, deviceData, deviceCount);

    // Test result
    bool correct = true;
    float correctResult = 1.f * 3.f;
    for (int i=0 ; i< hostStructure->Size() ; i++)
        if (hostResult[i] != correctResult)
            correct = false;
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    // ~!

    return 0;
}

