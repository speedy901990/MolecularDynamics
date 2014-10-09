#include "GpuThread.h"

GpuThread::GpuThread() {

}

GpuThread::GpuThread(GpuKernel *inputKernel, int threadId):kernel(inputKernel), tid(threadId) {
 
}
