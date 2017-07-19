#! /usr/bin/python

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
 # The testing scripts are written in Python Version 3.0 release.
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

import re
import os
import sys
import glob
import time
import string
import socket
import argparse
import networkx as nx
from subprocess import call

def _logerr(log, msg):
    sys.stderr.write("error: %s"% (msg))
    sys.stderr.flush()
    log.write("error: %s"% (msg))
    log.flush()
## end _logerr

def _logmsg(log, msg):
    sys.stdout.write("%s"% (msg))
    sys.stdout.flush()
    log.write("%s"% (msg))
    log.flush()
## end _logmsg()

def _call(cmd, log=sys.stdout):
    ret = call(cmd, shell=True)
    if ret != 0:
        if ret < 0:
            _logerr(log, "killed by signal %d\n"% (ret))
        else:
            _logerr(log, "%s command failed with return code %d\n"% (cmd, ret))
        ## end if
        ##sys.exit()
    ## end if ret
    return ret
## end _call()

def cmd_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('input')

    g.add_argument('-bt', '--build-type', nargs='+', required=False, type=str,
                   default=['READER_BIN_HEAP', 'READER_FIB_HEAP', \
                            'READER_BIN_HEAP_PAR', 'READER_FIB_HEAP_PAR'] )
    g.add_argument('-c', '--arg-cmd', nargs='+', required=False, type=str,
                   default=['el'], choices=['dw', 'el', \
                   'erickson'])
    g.add_argument('--print', action='store_true')
    g.add_argument('--list', action='store_true')
    return parser
## end cmd_parser()

def print_sysdetails(log):
    _logmsg(log, 'system details:\n')
    _logmsg(log, "date: %s\n"% (time.strftime("%d/%m/%y")))
    _logmsg(log, "time: %s\n"% (time.strftime("%H:%M:%S")))
    _logmsg(log, "host: %s\n"% (socket.gethostname())) 
    _logmsg(log, "\n")
## end print_sysdetails()

def verify_graph(edgelist, terminals):
    G = nx.Graph()
    for edge in edgelist:
        e = edge.split(" ")
        G.add_edge(int(e[0]), int(e[1]))
    #end for
    if(nx.is_connected(G) == False):
        return -1
    terminalList = [int(q) for q in terminals.split(" ")]
    for q in terminalList:
        if q not in G.nodes():
            return q
    #end for
    return 0
#end verify_graph()
#*****************************************************************************
def _solution(line):
    #regex = re.compile(r'solution: \[(.*)\]', re.M | re.L )
    #return regex.search(line).group(1).strip()
    return re.findall(r'"([^"]*)"', line)
#end _solution()

def _terminals(line):
    regex = re.compile(r'terminals:(.*)', re.M | re.L )
    return regex.search(line).group(1).strip()
#end _terminals()

def _input(line):
    regex = re.compile(r'(.*)n = (.*), m = (.*), k = (.*), cost = (.*) \[(.*)ms\] {peak:(.*)GiB} {curr:(.*)GiB}',  re.M | re.I )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(2, 9)))
#end _input()

def _erickson(line):
    regex = re.compile('erickson: \[zero:(.*)ms\] \[kernel:(.*)ms (.*)GiB/s\] done. \[(.*)ms\] \[cost:(.*)] {peak:(.*)GiB} {curr:(.*)GiB}', re.M | re.L)
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 8)))
#end _erickson()

def _erickson_listsolution(line):
    regex = re.compile('erickson: \[zero:(.*)ms\] \[kernel:(.*)ms (.*)GiB/s\] \[traceback: (.*)ms\] done. \[(.*)ms\] \[cost:(.*)] {peak:(.*)GiB} {curr:(.*)GiB}', re.M | re.L)
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 9)))
#end _erickson_listsolution()

def parse_file(input_, listsolution):
    edgelist = []
    with open(input_, 'r') as infile:
        for line in infile:
            if line.startswith('input'):
                n, m, k, inCost, inTime, inPeak, inCurr = _input(line)
            elif line.startswith('erickson'):
                if listsolution:
                    eZero, eKernel, eKernelBw, etraceback, eTotal, eCost, ePeak, eCurr = _erickson_listsolution(line)
                else:
                    eZero, eKernel, eKernelBw, eTotal, eCost, ePeak, eCurr = _erickson(line) 
                #end if
            elif line.startswith('solution'):
                edgelist = _solution(line)
            elif line.startswith('terminals'):
                terminals = _terminals(line)
            #end if
        #end for
    #end fopen
    return inCost, eCost, terminals, edgelist
#end of parse_file()

def main():
    parser = cmd_parser()
    opts = vars(parser.parse_args())
    buildtypeList = opts['build_type']
    argcmdList = opts['arg_cmd']
    list_  = opts['list']
    print_ = opts['print']

    globalfail = False

    sys.stderr.write('invoked as: ')
    for i in range(0, len(sys.argv)):
        sys.stderr.write(" %s"% (sys.argv[i]))
    sys.stderr.write('\n\n')

    cwd = os.getcwd()
    exeDir = cwd + "/../reader"
    verifyDir = "%s/verifyDir"% (cwd)
    testsetDir = cwd + "/../testset/small-instances"
    datetime = "%s_%s"% (time.strftime("%d-%m-%y"), time.strftime("%H-%M-%S"))

    for build in buildtypeList:
        os.chdir(exeDir)
        cmd = "make %s"% (build)
        ret = _call(cmd, sys.stderr)
        os.chdir(cwd)
        if ret:
            sys.stderr.write("error: build <%s> failed\n"% (build))
            sys.exit()
        #end if


        cmd = "mkdir -p %s"% (verifyDir)
        ret = _call(cmd, sys.stderr)
        if ret:
            sys.stderr.write("error: failed to create <%s> directory\n"% \
                             (verifyDir))
            sys.exit()
        #end if

        cmd = "cp %s/%s %s"% (exeDir, build, verifyDir)
        ret = _call(cmd, sys.stderr)
        if ret:
            sys.stderr.write("error: failed to copy build <%s> \n"% (build))
            sys.exit()
        #end if
        
        failed = False
        for argcmd in argcmdList:
            logfile = "%s/run_%s_%s_%s.log"% \
                      (verifyDir, build, argcmd, datetime)
            outfile = "%s/verify_%s_%s_%s.out"% \
                      (verifyDir, build, argcmd, datetime)
            log = open(logfile, 'w')
            print_sysdetails(log)

            _logmsg(log, "cmd-line args:\n")
            _logmsg(log, "build: %s\n"% (build))
            _logmsg(log, "arg cmd: %s\n"% (argcmd))
            _logmsg(log, "\n")

            exe = "%s/%s"% (verifyDir, build)

            os.chdir(testsetDir)
            testfileList = glob.glob(os.path.join('*', '*.stp'))

            for testfile in testfileList:
                if list_:
                    cmd = "%s -in %s -%s -list 1>%s 2>%s"% \
                           (exe, testfile, argcmd, outfile, outfile)
                else:
                    cmd = "%s -in %s -%s 1>%s 2>%s"% \
                           (exe, testfile, argcmd, outfile, outfile)
                # end if
                if print_:
                    sys.stdout.write(cmd)
                    sys.stdout.write('\n\n')
                else:
                    _call(cmd, log)
                    _logmsg(log, "STP instance: %s "% (testfile))
                    if 'ERROR' in open(outfile).read():
                        _logerr(log, '\n\n****** error: testing failed ****\n\n')
                        failed = True
                        globalfail = True
                    elif('assert' in open(outfile).read() or 'Assert' in open(outfile).read()):
                        _logerr(log, '\n\n****** assert: testing failed ***\n\n')
                        failed = True
                        globalfail = True
                    else:
                        inCost, dwCost, terminals, edgelist= parse_file(outfile, list_)
                        if(inCost != dwCost):
                            _logmsg(log, "fail inCost = %d dwCost = %d\n"%\
                                          (incost, dwCost))
                        elif(list_):
                            ret = verify_graph(edgelist, terminals)
                            if(ret == 0):
                                _logmsg(log, "pass\n")
                            elif(ret == -1):
                                _logmsg(log, "fail, graph is disconnected\n")
                            else:
                                _logmsg(log, "fail, %d terminal not covered\n"% (ret))
                            #end if
                        else:
                            _logmsg(log, "pass\n")
                        #end if
                    #end if
                ## end if print_
            ## end for testfile
            cmd = "rm -f %s"% (outfile)
            _call(cmd)
            if failed:
                _logerr(log, "\n*********************************************\n"\
                             "FAIL\nFAIL\nFAIL"\
                             "\n*********************************************\n")
            else:
                _logmsg(log, "\n*********************************************\n"\
                             "PASS\nPASS\nPASS"\
                             "\n*********************************************\n")
            #end if
        ## end for argcmd
    ## end for build
    if globalfail:
        _logerr(log, "\n*********************************************\n"\
                     "GLOBAL:\n"\
                     "FAIL\nFAIL\nFAIL"\
                     "\n*********************************************\n")
    else:
        _logmsg(log, "\n*********************************************\n"\
                     "GLOBAL:\n"\
                     "PASS\nPASS\nPASS"\
                     "\n*********************************************\n")
    #end if
    os.chdir(cwd)
## end main()


if __name__ == "__main__":
    main()
