# Multistage-QW
Code for simulating a multi-stage quantum walk

This work is based on the paper "Finding spin glass ground states using quantum walks" by Adam Callison (https://doi.org/10.1088/1367-2630/ab5ca2), but extends it to allow more than 2 stages.

It uses the database mentioned in that paper, https://doi.org/10.15128/r21544bp097, which contains 10k spin glass instances to allow for reproducibility.

I recommend compiling MultiQW.cpp with the `-O3` and `-march=native` flags, note that it requires `eigen3` and `vcl2` to be installed. The resulting file takes 5 inputs, the parameters n and m, the path to the file containing 
the problem instances, the start point for reading the instances and the number of instances to read and process. The last two parameters are used by `run.sh` to automatically spawn multiple threads for parallelisation. 

This is a simplified version of the original code, that only implements short-time averages but in a highly optimised implementation. There are two choices for heuristic, the original heuristic by Adam Callison, 
and a new heuristic derived by me. They both perform similarly, but the new heuristic doesn't rely on known statistics of spin glasses so is preferred as it should generalise to other problem types.
In principle, a classical algorithm like simulated annealing could be used to estimate the energy spread too, but this hasn't been implemented yet. In a handful of tests done, simulated annealing always found
the exact ground state. A more accurate estimate of the ground state doesn't seem to benefit the algorithm much though.

A pre-print for the work will be available soon, and a link will be added here when it is.

Currently the code spits out a unique file for each (n,m) pair, which is a bit messy. These files contain 2000 floating point values, describing the probability of finding the ground state for that problem 
and number of stages.
