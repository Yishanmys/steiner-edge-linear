##
 # This file is part of an experimental software implementation of the
 # Erickson-Monma-Veinott algorithm for solving the Steiner problem in graphs.
 # The algorithm runs in edge-linear time and the exponential complexity is
 # restricted to the number of terminal vertices.
 #
 # This software was developed as part of my master thesis work 
 # "Scalable Parameterised Algorithms for two Steiner Problems" at Aalto
 # University, Finland.
 #
 # The source code is configured for a gcc build for Intel
 # microarchitectures. Other builds are possible but require manual
 # configuration of the 'Makefile'.
 #
 # The source code is subject to the following license.
 #
 # Copyright (c) 2017 Suhas Thejaswi
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 #/



MAKE = make
CC = gcc
CFLAGS = -O3 -Wall -march=native -std=c99 -fopenmp

all: gen-unique

gen-unique: gen-unique.c ffprng.h
	$(CC) $(CFLAGS) -o gen-unique gen-unique.c -lm

.PHONY: gen-unique
clean:
	rm -f *~ gen-unique
