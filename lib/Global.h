#ifndef __GLOBAL_H__
#define	__GLOBAL_H__

#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "AtomsStructure.h"
#include "CudaHelpers.h"
#include "helper_functions.h"
using namespace std;

/*ostream& operator<< (ostream& stm, AtomsStructure& s){
    for (int i=0 ; i<s.Size() ; i++){
    	stm << "X=" << s.x[i] << " Y=" << s.y[i] << " Z=" << s.z[i] << endl;
    }
    return stm;
}
*/

#endif	/* __GLOBAL_H__ */

