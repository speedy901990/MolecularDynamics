#ifndef GPU_DISPLAY_H
#define GPU_DISPLAY_H

// Utilities and timing functions
#include "helper_functions.h"
// CUDA helper functions
#include "helper_cuda.h"     
#include "helper_cuda_gl.h"  

#include <vector_types.h>

#include "Global.h"
#include "Structure.h"
#include "GpuKernel.h"

#define MAX(a,b) ((a > b) ? a : b)
class GpuKernel;

class GpuDisplay {
 public:
  static GpuDisplay * instance();
  int init(int argc, char ** argv, Structure * &structure);
  void runAnimation(GpuKernel * pKernel);

 private:
  static GpuDisplay * pInstance;
  // constants
  const float MAX_EPSILON_ERROR = 10.0f;
  const float THRESHOLD = 0.30f;
  const float REFRESH_DELAY = 10; //ms
  unsigned int window_width;
  unsigned int window_height;
  unsigned int mesh_width;
  unsigned int mesh_height;
  unsigned int mesh_depth;
  // vbo variables
  GLuint vbo;
  struct cudaGraphicsResource *cuda_vbo_resource;
  void *d_vbo_buffer;
  float g_fAnim;
  // mouse controls
  int mouse_old_x;
  int mouse_old_y;
  int mouse_buttons;
  float rotate_x;
  float rotate_y;
  float translate_x;
  float translate_y;
  float translate_z;
  StopWatchInterface *timer;
  // Auto-Verification Code
  int fpsCount;        // FPS count for averaging
  int fpsLimit;        // FPS limit for sampling
  int g_Index;
  float avgFPS;
  unsigned int frameCount;
  unsigned int g_TotalErrors;
  bool g_bQAReadback;

  Structure * structure;
  GpuKernel * kernel;

  GpuDisplay();
  void operator=(GpuDisplay const&);

  int initGL(int argc, char ** argv);
  void runCuda(struct cudaGraphicsResource **vbo_resource);
  void launch_kernel(float4 *pos, float time);

  static void displayWrapper();
  static void keyboardWrapper(unsigned char key, int x, int y);
  static void motionWrapper(int x, int y);
  static void mouseWrapper(int button, int state, int x, int y);
  static void cleanupWrapper();
  static void timerEventWrapper(int value);

  void display();
  void keyboard(unsigned char key, int x, int y);
  void motion(int x, int y);
  void mouse(int button, int state, int x, int y);

  void cleanup();
  void timerEvent(int value);
  void computeFPS();
  void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
  void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
};

#endif /* GPU_DISPLAY_H */
