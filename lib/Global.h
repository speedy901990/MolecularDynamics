/* 
 * File:   Global.h
 * Author: speedy
 *
 * Created on October 5, 2013, 8:20 AM
 */

#ifndef GLOBAL_H
#define	GLOBAL_H

#include <iostream>
#include "Structure.h"
using namespace std;

ostream& operator<< (ostream& stm, Atom& s){
    stm << "X=" << s.X() << " Y=" << s.Y() << " Z=" << s.Z() << endl;
}

#endif	/* GLOBAL_H */

