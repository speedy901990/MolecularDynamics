#ifndef GLOBAL_H
#define	GLOBAL_H

#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "Simulation.h"
//#include "AtomsStructure.h"

//#include "CudaHelpers.h"
#include "helper_functions.h"
using namespace std;

/*ostream& operator<< (ostream& stm, AtomsStructure& s){
    for (int i=0 ; i<s.Size() ; i++){
    	stm << "X=" << s.x[i] << " Y=" << s.y[i] << " Z=" << s.z[i] << endl;
    }
    return stm;
}
*/

#endif	/* GLOBAL_H */

