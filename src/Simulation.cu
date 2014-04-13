#include <iostream>
#include "Simulation.h"
using namespace std;

Simulation::Simulation() {}

Simulation * Simulation::pInstance = NULL;

Simulation * Simulation::instance() {
  if (!pInstance)
    pInstance = new Simulation();
}

int Simulation::perform() {
  cout << "Simulation done!" << endl;
}

int Simulation::setInitParams() {

}

int Simulation::loadStructure(Structure * structure) {

}
