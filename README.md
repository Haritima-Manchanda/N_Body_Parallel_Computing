Converts the serial implementation of a solution to the nbody problem (nbody.c) to a parallelized version (nbody.cu) using CUDA.

The program greatly increases performance by utilizing the processing power of a GPU.

# The n-body problem:
The n-body problem is a classical computation model where we have a system with n objects in it (like our solar system), each object having a mass, a velocity in 3 dimensions, and a position in 3 dimensions.  The idea is to track/compute over time the movement and speed of all of the objects in the system, taking into account the gravitational interaction between them.  This is an O(n^2) algorithm, as for each object, we have to sum up all of the effects of the other (n-1) objects.

# PROBLEM STATEMENT:
The n-body simulator that I am providing you models our solar system (sun + 8 planets (sorry pluto)) then adds some number of random asteroids to the system.  These asteroids can appear anywhere in the system and can have a mass anywhere from near 0 to 938x10^18 kg which is the approximate size of the asteroid Ceres.  The file config.h allows you to modify the parameters of the simulation. Of particular interest is NUMASTEROIDS.  As an O(n^2) algorithm, the initial value of 10 will not be a problem on a CPU, but if we take that up to 1000, it will take minutes.  There are well over 1,000,000 known asteroids in our solar system, so we need a better solution. Therefore, we need to parallelize it.
