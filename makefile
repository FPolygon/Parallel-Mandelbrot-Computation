# Compiler settings
CXX := mpic++
CXXFLAGS := -std=c++11 -Wall -Wextra -fopenmp
LDFLAGS := -lm

# Executable name
EXE := mandelbrot

# Source files
SRCS := mandelbrot_mpi.cc
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(EXE)

# Link object files to create executable
$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(EXE) $(OBJS)