#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "GpuDisplay.h"
#include "Log.h"
#include "CudaHelpers.h"

GpuDisplay::GpuDisplay() {

}

GpuDisplay::~GpuDisplay() {

}

int GpuDisplay::init() {
  d_vbo_buffer = NULL;
  g_fAnim = 0.0;
  
  // mouse controls
  mouse_buttons = 0;
  rotate_x = 0.0;
  rotate_y = 0.0;
  translate_z = -3.0;
  //timer = NULL;

  // Auto-Verification Code
  fpsCount = 0;        // FPS count for averaging
  fpsLimit = 1;        // FPS limit for sampling
  g_Index = 0;
  avgFPS = 0.0f;
  frameCount = 0;
  g_TotalErrors = 0;
  g_bQAReadback = false;
}
