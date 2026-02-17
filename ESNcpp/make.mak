CXX      = g++
CXXFLAGS = -O3 -std=c++20 -fopenmp -march=native
INCLUDES = -I eigen-master
LDFLAGS  = -static -static-libgcc -static-libstdc++

# La clave: Listamos los archivos objeto (.o)
OBJS = Res_test.o ESN.o Math.o
TARGET = test.exe

all: $(TARGET)

# Une los .o para crear el ejecutable
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(TARGET)

# Compila cada .cpp a .o de forma independiente
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.o $(TARGET)

.PHONY: all run clean