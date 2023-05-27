FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody  
