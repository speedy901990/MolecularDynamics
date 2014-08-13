#ifndef ATOM_H
#define ATOM_H

struct Position {
  float x, y, z;
};

class Atom {
 public:
  Atom();
  virtual ~Atom();
  Position pos;
  Position initPos;
  float force;
  float initForce;
  float acceleration;
  enum Status{ REGULAR, BOUNDARY }; // status
  Status status;
  bool fixed;

  int init(float x, float y, float z, float force, Status status, bool fixed = false);
};

#endif /* ATOM_H */
