#ifndef ATOMSSTRUCTURE_H
#define	ATOMSSTRUCTURE_H

#include <cuda_runtime.h>
#include "helper_functions.h"

class AtomsStructure {
public:
    float *x, *y, *z;
    float *result;
	int deviceID;
    int iterN;
    AtomsStructure();
    AtomsStructure(int size);
    AtomsStructure(const AtomsStructure& orig);
    virtual ~AtomsStructure();
    int Size();
    int DeviceID();

private:
	int size;
};

#endif	/* ATOMSSTRUCTURE_H */

