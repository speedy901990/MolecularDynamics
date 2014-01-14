/* 
 * File:   Atom.h
 * Author: speedy
 *
 * Created on September 4, 2013, 10:29 PM
 */

#ifndef ATOM_H
#define	ATOM_H

class Atom {
public:
    int x, y, z;
    int X();
    int Y();
    int Z();
    Atom();
    Atom(const Atom& orig);
    virtual ~Atom();
    
private:
    
};

#endif	/* ATOM_H */

