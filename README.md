# Multistage-QW
Code for simulating a multi-stage quantum walk

This work is based on the paper "Finding spin glass ground states using quantum walks" by Adam Callison (https://doi.org/10.1088/1367-2630/ab5ca2), but extends it to allow more than 2 stages.

It requires the database mentioned in that paper, https://doi.org/10.15128/r21544bp097, which contains 10k spin glass instances to allow for reproducibility.

There's a variety of choices for the calculation to be performed. Heuristic gamma values can be used (which I derive in my notes), or a classical optimiser is implemented for finding optimal gamma values (quickly becomes intractable).
The infinite time average over all stages can be calculated, or short time averages can be done (where I derive a reasonable timescale in my notes, this disagrees with the Callison paper's numerically derived timescale).
It is also possible to plot the energy (w.r.t the graph Hamiltonian) and log-probability over the duration of the quantum walk, which shows them both converging from `-n` to `0` for enough stages.

I'll add a link to the pre-print when it becomes available.

These are currently unfinished, but outline the formulae for infinite time averages, heuristic gamma values, and short average time scale. These notes also show that a polynomial scaling algorithm can be derived, which follows from the fact that the exponential scaling exponent seems to have a polynomial relationship with the number of stages in the quantum walk.

Currently the code spits out a unique file for each (n,m) pair and each type of calculation, so I hope to find some way to put everything in one database before uploading it here as the number of files is messy.
The same goes for the database mentioned above, it contains 10k individual files which can be inconvenient at times, so I'll try and provide a single file version.
