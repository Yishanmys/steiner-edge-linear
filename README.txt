Overview
--------
This software repository contains a parallel implementation of the 
Erickson-Monma-Veinott algorithm for solving the Steiner problem in graphs, 
and by reduction, the group Steiner problemi in graphs. It is a parameterised
algorithm, which runs in edge-linear time and the exponential complexity is
restricted to the number of terminals. The software is written in C programming
language and OpenMP API for parallelisation.

This software was developed as part of my master thesis work 
"Scalable Parameterised Algorithms for two Steiner Problems" at Aalto
University, Finland.

The source code is configured for a gcc build for Intel
microarchitectures. Other builds are possible but require manual
configuration of the 'Makefile'.

The source code is subject to MIT license, see 'LICENSE' for details.

Building
--------
Use GNU make to build the software.

Our implementation makes use of preprocessor directives to enable conditional
compilation for generating the binaries specific to single or multi threaded
variants; optimal cost or optimal solution variants of the software. In
addition to this, resource tracking can be enabled using 'TRACK_RESOURCES' 
compilation flag. 

Check 'Makefile' for building the software.

Testing
-------
For testing use 'verify.py' script provided along with the software. The script 
is written in Python Version 3.0 release and it uses 'networkx' python package.
The test instances used for the purpose  of testing this software are available
in the 'testset' directory.

Use the command-line options to verify the optimal cost and optimal solution 
of the test instances.

Optimal cost: ./verify.py -bt reader-el
Optimal solution: ./verify.py -bt reader-el --list

Experiments
-----------
Use 'report.py' script provided along with the software to perform the 
experiments to verify the scalability of the software. The software is written
in Python Version 3.0 release.

usage: report.py [-h] [-run] [-parse] [-b] [-bt BUILD_TYPE [BUILD_TYPE ...]]
                 [-m [{cpu-corei5,cpu-hsw,cpu-hsw-largemem,cpu-hsw-hugemem}]]
                 [-e [{reader-el}]] [-rep [REPORT_DIR]]
                 [-arg [{dijkstra,erickson}]] [-list]
                 [-g [{gen-unique,gen-count,gen-natural}]]
                 [-gt [{regular}]]
                 [-ft [{bin,ascii}]] [-n NODES [NODES ...]] [-d DEG [DEG ...]]
                 [-k TERMINALS [TERMINALS ...]] [-al [ALPHA]] [-w [WEIGHT]]
                 [-ew [EDGE_WEIGHT]] [-r [REPEATS]] [-gr [GRAPH_REPEATS]]
                 [-vr [VERTEX_REPEATS]] [-s [SEED]] [-p]
                 [-gp GNUPLOT_FILE [GNUPLOT_FILE ...]] [-err] [-bin] [-tar]
                 [-imp] [-pr] [-log [{r,w,a}]]

optional arguments:
  -h, --help            show this help message and exit


Usage
-----
./reader-el-in <input graph> <arguments>

arguments:
    -seed : seed value
    -el : Erickson-Monma-Veinott algorithm
    -dijkstra : Dijkstra single source shortest path
    -list : Output Steiner tree

./reader-el -in b01.stp -el -list
invoked as: ./reader-el -in b01.stp -el -list
no random seed given, defaulting to 123456789
random seed = 123456789
input: n = 50, m = 63, k = 9, cost = 82 [0.10 ms] {peak: 0.00GiB} {curr: 0.00GiB}
terminals: 48 49 22 35 27 12 37 34 24
root build: [zero: 2.54 ms] [pos: 0.01 ms] [adj: 0.01 ms] [term: 0.00 ms] done. [2.59 ms] {peak: 0.00GiB} {curr: 0.00GiB}
command: Erickson-Monma-Veinott
erickson: [zero: 0.13 ms] [kernel: 5.62 ms 0.93GiB/s] [traceback: 0.00 ms] done. [5.85 ms] [cost: 82] {peak: 0.00GiB} {curr: 0.00GiB}
solution: ["24 28", "28 18", "18 43", "43 22", "22 12", "22 41", "41 49", "41 37", "22 20", "20 48", "20 35", "20 27", "27 34"]
command done [5.99 ms]
grand total [9.83 ms] {peak: 0.00GiB}
host: cs-119
build: edge-linear kernel, multi-threaded, binary heap
list solution: true
num threads: 4
compiler: gcc 5.4.0


Input graphs
------------
The input graphs should be in DIMACS STP format. Our implementation accepts the
STP file only in ASCII files and the characters must be in lower-case.
