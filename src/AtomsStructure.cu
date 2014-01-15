#include "AtomsStructure.h"

AtomsStructure::AtomsStructure(){
    deviceID = 0;
    size = 3;
    float initValue = 1.f;
    x = new float[size];
    y = new float[size];
    z = new float[size];

    for (int i=0 ; i<size ; i++){
        x[i] = initValue;
        y[i] = initValue;
        z[i] = initValue;
    }
}

AtomsStructure::AtomsStructure(int size) {
    deviceID = 0;
    this->size = size;
    float initValue = 1.f;
    x = new float[size];
    y = new float[size];
    z = new float[size];
    result = new float[size];

    for (int i=0 ; i<size ; i++){
        x[i] = initValue;
        y[i] = initValue;
        z[i] = initValue;
    }
}

AtomsStructure::AtomsStructure(const AtomsStructure& orig) {
}

AtomsStructure::~AtomsStructure() {
}

int AtomsStructure::Size() {
    return size;
}
