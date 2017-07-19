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
 ##


MAKE = make
CC = gcc 
CFLAGS = -Wall -march=native -std=c99 -fopenmp -DTRACK_RESOURCES

SOURCE = reader-el.c

EXE = reader-el \
	READER_DEFAULT \
	READER_BIN \
	READER_BIN_PAR \
	READER_BIN_OPT \
	READER_BIN_OPT_PAR \
	READER_FIB \
	READER_FIB_PAR \
	READER_FIB_OPT \
	READER_FIB_OPT_PAR \
	READER_BIN_DIJK

all: $(EXE)

reader-el: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -DBUILD_PARALLEL -DTRACK_OPTIMAL -o $@ $< -lm

READER_DEFAULT: $(SOURCE)
	$(CC) $(CFLAGS) -g -DBIN_HEAP -DBUILD_PARALLEL -DTRACK_OPTIMAL -DDEBUG -o $@ $< -lm

READER_BIN: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -o $@ $< -lm

READER_BIN_PAR: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -DBUILD_PARALLEL -o $@ $< -lm

READER_BIN_OPT: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -DTRACK_OPTIMAL -o $@ $< -lm

READER_BIN_OPT_PAR: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -DTRACK_OPTIMAL -DBUILD_PARALLEL -o $@ $< -lm

READER_FIB: $(SOURCE)
	$(CC) $(CFLAGS) -DFIB_HEAP -o $@ $< -lm

READER_FIB_PAR: $(SOURCE)
	$(CC) $(CFLAGS) -DFIB_HEAP -DBUILD_PARALLEL -o $@ $< -lm

READER_FIB_OPT: $(SOURCE)
	$(CC) $(CFLAGS) -DFIB_HEAP -DTRACK_OPTIMAL -o $@ $< -lm

READER_FIB_OPT_PAR: $(SOURCE)
	$(CC) $(CFLAGS) -DFIB_HEAP -DTRACK_OPTIMAL -DBUILD_PARALLEL -o $@ $< -lm

READER_BIN_DIJK: $(SOURCE)
	$(CC) $(CFLAGS) -DBIN_HEAP -DDIJKSTRA_BENCHMARK -o $@ $< -lm

READER_FIB_DIJK: $(SOURCE)
	$(CC) $(CFLAGS) -DFIB_HEAP -DDIJKSTRA_BENCHMARK -o $@ $< -lm

.PHONY: $(EXE)

clean:  
	rm -f *.o *.a *~ 
	rm -f $(EXE)
