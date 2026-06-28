# =============================================================================
# make.mak  —  Build the two ESN engines on Windows (MSYS2 / mingw64).
#
# One-time setup (in an "MSYS2 MINGW64" shell):
#     pacman -Syu                                  # update, may ask to reopen
#     pacman -S mingw-w64-x86_64-gcc \
#               mingw-w64-x86_64-eigen3 \
#               mingw-w64-x86_64-make
#
# Build (from this folder, in an MSYS2 MINGW64 shell):
#     mingw32-make -f make.mak              # builds esn_run.exe and esn_ipc.exe
#     mingw32-make -f make.mak clean
#
# NOTE: The notebook compiles these automatically via subprocess, so you only
#       need this makefile if you want to build the binaries by hand.
# =============================================================================

CXX      := g++
# Eigen ships header-only; on mingw64 it lives under /mingw64/include/eigen3.
EIGEN    := /mingw64/include/eigen3
CXXFLAGS := -std=c++20 -O3 -fopenmp -I$(EIGEN) \
            -Wno-deprecated-declarations -Wno-deprecated-enum-enum-conversion

# Shared translation units (compiled once, linked into both engines).
COMMON_OBJ := ESN.o Math.o Utils.o
HEADERS    := ESN.h Task.h Config.h

.PHONY: all clean
all: esn_run.exe esn_ipc.exe

# ---- shared objects ---------------------------------------------------------
ESN.o: ESN.cpp ESN.h
	$(CXX) $(CXXFLAGS) -c ESN.cpp -o $@

Math.o: Math.cpp
	$(CXX) $(CXXFLAGS) -c Math.cpp -o $@

Utils.o: Utils.cpp
	$(CXX) $(CXXFLAGS) -c Utils.cpp -o $@

# ---- engines ----------------------------------------------------------------
esn_run.exe: esn_run.cpp $(COMMON_OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) esn_run.cpp $(COMMON_OBJ) -o $@

esn_ipc.exe: esn_ipc.cpp $(COMMON_OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) esn_ipc.cpp $(COMMON_OBJ) -o $@

clean:
	rm -f *.o esn_run.exe esn_ipc.exe
