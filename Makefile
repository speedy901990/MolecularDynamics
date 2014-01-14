NVCC = nvcc
NVCCFLAGS = 
LDFLAGS = 
INCLUDES = -I./$(LIBDIR)
OBJDIR = obj
SRCDIR = src
LIBDIR = lib
CPPS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(addprefix $(OBJDIR)/,$(notdir $(CPPS:.cpp=.o)))
NAME = MolecularDynamics

all: MolecularDynamics

MolecularDynamics: $(OBJS)
	$(NVCC) $(LDFLAGS) -o $(NAME) $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean
clean: 
	rm $(OBJDIR)/*.o 
	rm -rf $(OBJDIR)
	rm $(NAME)