NVCC = nvcc
CC = g++
NVCCFLAGS = -w -g -arch=sm_21
LDFLAGS = -w -g -L/usr/local/cuda/lib64
INCLUDES = -I./$(HDRDIR) -I/usr/local/cuda/include
LIBS = -lcudart -lpthread -lGL -lGLU -lGLEW -lglut
OBJDIR = obj
SRCDIR = src
HDRDIR = hdr
CPPS = $(wildcard $(SRCDIR)/*.cpp)
CUS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(addprefix $(OBJDIR)/,$(notdir $(CPPS:.cpp=.o)))
OBJS_CU = $(addprefix $(OBJDIR)/,$(notdir $(CUS:.cu=.o)))
NAME = MolecularDynamics

all: MolecularDynamics

MolecularDynamics: $(OBJS) $(OBJS_CU)
	$(CC) $(LDFLAGS) $(LIBS) -o $(NAME) $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(LDFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean
clean: 
	rm $(OBJDIR)/*.o 
	rm -rf $(OBJDIR)
	rm $(NAME)
