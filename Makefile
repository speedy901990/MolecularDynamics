NVCC = nvcc
CC = g++
NVCCFLAGS = -w -g
LDFLAGS = -w -g
INCLUDES = -I./$(HDRDIR)
OBJDIR = obj
SRCDIR = src
HDRDIR = hdr
CPPS = $(wildcard $(SRCDIR)/*.cpp)
CUS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(addprefix $(OBJDIR)/,$(notdir $(CPPS:.cpp=.o)))
CU_OBJS = $(addprefix $(OBJDIR)/,$(notdir $(CUS:.cu=.o)))
NAME = MolecularDynamics

all: MolecularDynamics

MolecularDynamics: $(OBJS)
	$(CC) $(LDFLAGS) -o $(NAME) $^

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
