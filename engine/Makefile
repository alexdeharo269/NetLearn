# Build the five ESN/MC programs (Linux / macOS).
#   make EIGEN=/path/to/eigen-master        # or put eigen-master next to the sources
#   make clean
# Note: C++17 is required — Eigen's LDLT does not compile under C++20 on recent g++.
EIGEN    ?= ./eigen-master
CXX      ?= g++
CXXFLAGS  = -std=c++17 -O2 -fopenmp -I$(EIGEN) -w
PROGS     = esn_mc esn_surrogates esn_bio esn_ipc esn_trace

all: $(PROGS)

%: %.cpp esn.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(PROGS)

.PHONY: all clean
