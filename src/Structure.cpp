/* 
 * File:   Structure.cpp
 * Author: speedy
 * 
 * Created on October 5, 2013, 7:33 AM
 */

#include "Structure.h"

Structure::Structure() {
    const int size = 3;
    structure = new Atom**[size];
    for (int i=0 ; i<size ; i++) {
        structure[i] = new Atom*[size];
    }
    for (int i=0 ; i<size ; i++){
        for (int j=0 ; j<size ; j++) {
            structure[i][j] = new Atom[size];
        }
    }
    
    for (int i=0 ; i<size ; i++){
        for (int j=0 ; j<size ; j++) {
            for (int k=0 ; k<size ; k++){
                structure[i][j][k].x = size;              
                structure[i][j][k].y = size;
                structure[i][j][k].z = size;

            }
        }
    }
}

Structure::Structure(const Structure& orig) {
}

Structure::~Structure() {
}

