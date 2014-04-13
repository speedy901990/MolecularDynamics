NVCC = nvcc
GCC = g++
NVCCFLAGS = -w -g
LDFLAGS = 
INCLUDES = -I./$(LIBDIR)
OBJDIR = obj
SRCDIR = src
LIBDIR = lib
CPPS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(addprefix $(OBJDIR)/,$(notdir $(CPPS:.cu=.o)))
NAME = MolecularDynamics

all: MolecularDynamics

MolecularDynamics: $(OBJS)
	$(NVCC) $(LDFLAGS) -o $(NAME) $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean
clean: 
	rm $(OBJDIR)/*.o 
	rm -rf $(OBJDIR)
	rm $(NAME)