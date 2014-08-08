#include "GpuHandler.h"

GpuHandler::GpuHandler() {}

GpuHandler * GpuHandler::pInstance = NULL;

GpuHandler * GpuHandler::instance() {
  if (!pInstance)
    pInstance = new GpuHandler();

  return pInstance;
}


