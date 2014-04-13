struct Position {
  float x, y, z;
};

class Atom {
 public:
  Atom();
  Position pos;
  Position initPos;
  float force;
  float initForce;
  float acceleration;
  int status;
  bool fixed;

  int init(float x, float y, float z, float force, int status, bool fixed = false);
};

