/* 
 * File:   main.cpp
 * Author: speedy
 *
 * Created on September 4, 2013, 10:21 PM
 */

#include "Global.h"
/*
 * 
 */
int main(int argc, char** argv) {
    Structure *str = new Structure();
    
    for (int i=0 ; i<3 ; i++){
        for (int j=0 ; j<3 ; j++) {
            for (int k=0 ; k<3 ; k++){
                cout << str->structure[i][j][k];
            }
        }
    }
    
    return 0;
}

