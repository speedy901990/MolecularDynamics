/* 
 * File:   Structure.h
 * Author: speedy
 *
 * Created on October 5, 2013, 7:33 AM
 */

#ifndef STRUCTURE_H
#define	STRUCTURE_H

#include "Atom.h"

class Structure {
public:
    Atom ***structure;
    Structure();
    Structure(const Structure& orig);
    virtual ~Structure();
    
private:
};

#endif	/* STRUCTURE_H */

