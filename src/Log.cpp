#include "Log.h"

Log::Log() { }

Log * Log::pInstance = NULL;

Log * Log::instance() {
  if (!pInstance)
    pInstance = new Log();

  return pInstance;
}

void Log::toConsole(int err, string className, string functionName, int line, string notes) {
  string msg = handleError(err);
  cerr << "ERR: " << className << "::" << functionName << " at line " << line << endl
       << ">> " <<  msg <<  endl << "Notes: " << notes << endl;
}

void Log::toFile(int err, string className, string functionName, int line, string notes) {
  // TODO
}

string Log::handleError(int err) {
  string msg;
  switch (err){
  case INIT_ALREADY_COMPLETED:
    msg = "Init has already been completed. No need to do it again.";
    break;
  case E_CORRUPTED_STRUCTURE:
    msg = "Atoms structure does not exist or is corrupted, simulation cannot be done.";
    break;
  case E_FILE_NOT_FOUND:
    msg = "File not found.";
    break;
  case E_CONFIG_FILE_PARSE:
    msg = "File parsing error.";
    break;
  case E_GPU_PARAMS:
    msg = "GPU parameters were not initialized.";
    break;
  case E_PARAMS_NOT_INITIALIZED:
    msg = "Parameters were not initialized. For usage info hit try 'help' or '?'.";
    break;
  case I_DEVICE_NOT_SELECTED:
    msg = "No device was selected. For usage info hit try 'help' or '?'.";
    break;
  case E_INSUFFICIENT_DEVICES_LIMIT:
    msg = "Insufficient devices limit.";
    break;
  case E_NULL_PTR:
    msg = "Null pointer passed.";
    break;
  default:
    msg = "Unknown error";
  }
  return msg;
}

