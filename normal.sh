#!/usr/bin/env bash

# Remove all compiled programs before starting
rm blurOMP.out

# How many threads do you want to test from and too?
tmin=2
tmax=10

# How many chunks do you want to test from and too?
cmin=2
cmax=10

# How many results do you want?
runs=10

# Static, dynamic, guided, auto?
# The number is the code that OMP stores in its enum omp_sched_t custom variable type
# static = 1
# dynamic = 2
# guided = 3
# auto = 4
sch=static

# Recompile program
gcc -fopenmp -o blur.out blur.c

# Run programs

for z in $(seq 1 $runs);
do
  ./blurOMP.out
done


printf "\nfin\n"

# If you want to do only even numbers in a seq
# You can add an interval to the seq command as the second argument
# i.e
# seq 2 2 10
# will output:
# 2 4 6 8 10

# To put the output of the script into a file (as opposed to printing to the screen)
# Run `bash runner.sh >> outputfile.txt`
