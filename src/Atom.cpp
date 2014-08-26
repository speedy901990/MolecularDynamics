#include <iostream>
#include "Atom.h"
using namespace std;

Atom::Atom() {
  float initValue = 1;
  pos.x = initValue;
  pos.y = initValue;
  pos.z = initValue;
  initPos.x = initValue;
  initPos.y = initValue;
  initPos.z = initValue;
  this->force = initValue;
  initForce = initValue;
  this->status = REGULAR;
  this->fixed = false;
}

Atom::~Atom() {
  
}

int Atom::init(float x, float y, float z, float force, Status status, bool fixed) {
  pos.x = x;
  pos.y = y;
  pos.z = z;
  initPos.x = x;
  initPos.y = y;
  initPos.z = z;
  this->force = force;
  initForce = force;
  this->status = status;
  this->fixed = fixed;
  acceleration = 1;
}

Atom& Atom::operator=(const Atom &orgAtom) {
  if (&orgAtom == this)
    return *this;

  pos.x = orgAtom.pos.x;
  pos.y = orgAtom.pos.y;
  pos.z = orgAtom.pos.z;
  
  initPos.x = orgAtom.initPos.x;
  initPos.y = orgAtom.initPos.y;
  initPos.z = orgAtom.initPos.z;
  
  force = orgAtom.force;
  initForce = orgAtom.initForce;
  acceleration = orgAtom.acceleration;
  status = orgAtom.status;
  fixed = orgAtom.fixed;

  return *this;
}
