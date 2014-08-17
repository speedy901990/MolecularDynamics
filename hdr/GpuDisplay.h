#ifndef GPU_DISPLAY_H
#define GPU_DISPLAY_H

// OpenGL Graphics includes
//#include "GL/glew.h"
#include "GL/freeglut.h"

// Utilities and timing functions
#include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h
//#include "timer.h"               // timing functions

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check
#include "helper_cuda_gl.h"      // helper functions for CUDA/GL interop

//#include <vector_types.h>

#include "Global.h"

#define MAX(a,b) ((a > b) ? a : b)

class GpuDisplay {
 public:
  GpuDisplay();
  ~GpuDisplay();
  int init();

 private:
  // constants
  const float MAX_EPSILON_ERROR = 10.0f;
  const float THRESHOLD = 0.30f;
  const float REFRESH_DELAY = 10; //ms
  const unsigned int window_width  = 512;
  const unsigned int window_height = 512;
  const unsigned int mesh_width    = 256;
  const unsigned int mesh_height   = 256;

  // vbo variables
  GLuint vbo;
  struct cudaGraphicsResource *cuda_vbo_resource;
  void *d_vbo_buffer;
  float g_fAnim;

  // mouse controls
  int mouse_old_x, mouse_old_y;
  int mouse_buttons;
  float rotate_x, rotate_y;
  float translate_z;
  //StopWatchInterface *timer;

  // Auto-Verification Code
  int fpsCount;        // FPS count for averaging
  int fpsLimit;        // FPS limit for sampling
  int g_Index;
  float avgFPS;
  unsigned int frameCount;
  unsigned int g_TotalErrors;
  bool g_bQAReadback;
};

#endif /* GPU_DISPLAY_H */
