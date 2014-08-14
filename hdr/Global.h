#ifndef GLOBAL_H
#define	GLOBAL_H

#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <exception>
#include <string.h>
#include <typeinfo>
using namespace std;

enum {
  SUCCESS = 0,
  FAIL = -1,
  INIT_ALREADY_COMPLETED = -2,
  E_CORRUPTED_STRUCTURE = -3,
  E_FILE_NOT_FOUND = -4,
  E_CONFIG_FILE_PARSE = -5,
  E_GPU_PARAMS = -6,
  E_PARAMS_NOT_INITIALIZED = -7,
  I_DEVICE_NOT_SELECTED = -8,
  E_INSUFFICIENT_DEVICES_LIMIT = -9
};

/*ostream& operator<< (ostream& stm, AtomsStructure& s){
    for (int i=0 ; i<s.Size() ; i++){
    	stm << "X=" << s.x[i] << " Y=" << s.y[i] << " Z=" << s.z[i] << endl;
    }
    return stm;
}
*/

#endif	/* GLOBAL_H */

