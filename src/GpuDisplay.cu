// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "GpuDisplay.h"
#include "Log.h"
#include "CudaHelpers.h"

GpuDisplay::GpuDisplay() {}

GpuDisplay * GpuDisplay::pInstance = NULL;

GpuDisplay * GpuDisplay::instance() {
  if (!pInstance)
    pInstance = new GpuDisplay();

  return pInstance;
}

int GpuDisplay::init(int argc, char ** argv) {
  window_width  = 512;
  window_height = 512;
  mesh_width    = 256;
  mesh_height   = 256;

  d_vbo_buffer = NULL;
  g_fAnim = 0.0;
  
  // mouse controls
  mouse_buttons = 0;
  rotate_x = 0.0;
  rotate_y = 0.0;
  translate_z = -3.0;
  timer = NULL;

  // Auto-Verification Code
  fpsCount = 0;        // FPS count for averaging
  fpsLimit = 1;        // FPS limit for sampling
  g_Index = 0;
  avgFPS = 0.0f;
  frameCount = 0;
  g_TotalErrors = 0;
  g_bQAReadback = false;

  initGL(argc, argv);
}

int GpuDisplay::initGL(int argc, char ** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("Cuda GL Interop (VBO)");
  glutDisplayFunc(displayWrapper);
  glutKeyboardFunc(keyboardWrapper);
  glutMotionFunc(motionWrapper);
  glutTimerFunc(REFRESH_DELAY, timerEventWrapper,0);

  // initialize necessary OpenGL extensions
  glewInit();

  if (! glewIsSupported("GL_VERSION_2_0 "))
    {
      fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
      fflush(stderr);
      return false;
    }

  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);

  // viewport
  glViewport(0, 0, window_width, window_height);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

  SDK_CHECK_ERROR_GL();
  
  return SUCCESS;
}

void GpuDisplay::runCuda(struct cudaGraphicsResource **vbo_resource) {
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void GpuDisplay::launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time) {
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);
    // simple_vbo_kernel<<<1,1>>>(pos, mesh_width, mesh_height, time);
}

void GpuDisplay::runAnimation() {
  // register callbacks
  glutDisplayFunc(displayWrapper);
  glutKeyboardFunc(keyboardWrapper);
  glutMouseFunc(mouseWrapper);
  glutMotionFunc(motionWrapper);

  // create VBO
  createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

  // run the cuda part
  runCuda(&cuda_vbo_resource);

  // start rendering mainloop
  glutMainLoop();
  atexit(cleanupWrapper);
}

void GpuDisplay::displayWrapper() {
  pInstance->display();
}

void GpuDisplay::keyboardWrapper(unsigned char key, int x, int y) {
  pInstance->keyboard(key, x, y);
}

void GpuDisplay::motionWrapper(int x, int y) {
  pInstance->motion(x, y);
}

void GpuDisplay::mouseWrapper(int button, int state, int x, int y) {
  pInstance->mouse(button, state, x, y);
}

void GpuDisplay::cleanupWrapper() {
  pInstance->cleanup();
}

void GpuDisplay::timerEventWrapper(int value) {
  pInstance->timerEvent(value);
}

void GpuDisplay::display() {
  sdkStartTimer(&timer);

  // run CUDA kernel to generate vertex positions
  runCuda(&cuda_vbo_resource);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // set view matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, translate_z);
  glRotatef(rotate_x, 1.0, 0.0, 0.0);
  glRotatef(rotate_y, 0.0, 1.0, 0.0);

  // render from the vbo
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);

  glEnableClientState(GL_VERTEX_ARRAY);
  glColor3f(1.0, 0.0, 0.0);
  glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
  glDisableClientState(GL_VERTEX_ARRAY);

  glutSwapBuffers();

  g_fAnim += 0.01f;

  sdkStopTimer(&timer);
  computeFPS();
}

void GpuDisplay::computeFPS() {
  frameCount++;
  fpsCount++;

  if (fpsCount == fpsLimit) {
    avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    fpsCount = 0;
    fpsLimit = (int)MAX(avgFPS, 1.f);

    sdkResetTimer(&timer);
  }

  char fps[256];
  sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
  glutSetWindowTitle(fps);
}

void GpuDisplay::timerEvent(int value) {
  glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEventWrapper,0);
}

void GpuDisplay::cleanup() {
  sdkDeleteTimer(&timer);

  if (vbo) {
    deleteVBO(&vbo, cuda_vbo_resource);
  }
}

void GpuDisplay::keyboard(unsigned char key, int /*x*/, int /*y*/) {
  switch (key) {
  case (27) :
    exit(EXIT_SUCCESS);
    break;
  }
}

void GpuDisplay::mouse(int button, int state, int x, int y) {
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1<<button;
  }
  else if (state == GLUT_UP) {
    mouse_buttons = 0;
  }

  mouse_old_x = x;
  mouse_old_y = y;
}

void GpuDisplay::motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - mouse_old_x);
  dy = (float)(y - mouse_old_y);

  if (mouse_buttons & 1) {
    rotate_x += dy * 0.2f;
    rotate_y += dx * 0.2f;
  }
  else if (mouse_buttons & 4) {
    translate_z += dy * 0.01f;
  }

  mouse_old_x = x;
  mouse_old_y = y;
}

void GpuDisplay::createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
  assert(vbo);

  // create buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  // initialize buffer object
  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

  SDK_CHECK_ERROR_GL();
}

void GpuDisplay::deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res){
  // unregister this buffer object with CUDA
  cudaGraphicsUnregisterResource(vbo_res);

  glBindBuffer(1, *vbo);
  glDeleteBuffers(1, vbo);

  *vbo = 0;
}
