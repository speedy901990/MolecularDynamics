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
  this->status = initValue;
  this->fixed = false;
}

int Atom::init(float x, float y, float z, float force, int status, bool fixed) {
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
}
