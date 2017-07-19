#!/usr/bin/python

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
import time
import argparse
import itertools
import sys
import random
import string
import time
import glob
import traceback
import networkx as nx
from subprocess import call

# global variables
__RUN_EXP = False

#***************************************** parse output file ******************
def _random_seed(line):
    regex = re.compile(r'random seed = (.*)', re.M | re.I )
    return regex.search(line).group(1).strip()

def _input(line):
    regex = re.compile(r'(.*)n = (.*), m = (.*), k = (.*), cost = (.*) \[(.*)ms\] {peak:(.*)GiB} {curr:(.*)GiB}',  re.M | re.I )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(2, 9)))

def _root_build(line):
    regex = re.compile(r'root build:(.*)\[zero:(.*)ms\] \[pos:(.*)ms\] \[adj:(.*)ms\] \[term:(.*)ms\] done. \[(.*)ms\] {peak:(.*)GiB} {curr:(.*)GiB}', re.M | re.L )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(2, 9)))

def _oraclegpu(line):
    regex = re.compile(r'oracle: (.*) (.*)ms \[ *(.*)GiB (.*)GiB/s (.*)GHz (.*)GHz\] (.*) (.) -- (.*)', re.M | re.L )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 10)))

def _erickson(line):
    regex = re.compile('erickson: \[zero:(.*)ms\] \[kernel:(.*)ms (.*)GiB/s\] done. \[(.*)ms\] \[cost:(.*)] {peak:(.*)GiB} {curr:(.*)GiB}', re.M | re.L)
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 8)))

def _erickson_traceback(line):
    regex = re.compile('erickson: \[zero:(.*)ms\] \[kernel:(.*)ms (.*)GiB/s\] \[traceback: (.*)ms\] done. \[(.*)ms\] \[cost:(.*)] {peak:(.*)GiB} {curr:(.*)GiB}', re.M | re.L)
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 9)))

def _dijkstra(line):
    regex = re.compile(r'dijkstra:(.*)\[zero:(.*)ms\] \[hinsert:(.*)ms\] \[visit:(.*)ms\] \[total:(.*)ms (.*)GiB/s\] done. \[(.*)ms\] {peak:(.*)GiB}', re.M | re.L )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(2, 9)))

def _genunique(line):
    regex = re.compile(r'gen-unique \[(.*)\]: n = (.*), m = (.*), k = (.*), cost = (.*), seed = (.*)', re.M | re.L)
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 7)))

def _solution(line):
    return re.findall(r'"([^"]*)"', line)

def _terminals(line):
    regex = re.compile(r'terminals:(.*)', re.M | re.L )
    return regex.search(line).group(1).strip()

def _command_done(line):
    regex = re.compile(r'command done \[(.*)ms\]', re.M | re.L )
    return regex.search(line).group(1).strip()

def _grand_total(line):
    regex = re.compile(r'grand total \[(.*)ms\] {peak:(.*)GiB}(.*)', re.M | re.L )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 3)))

def _host(line):
    regex = re.compile(r'host:(.*)', re.M | re.L )
    return regex.search(line).group(1).strip()

def _build_type(line):
    regex = re.compile(r'build:(.*), (.*)', re.M | re.L )
    tokens = regex.search(line)
    return map(lambda string: string.strip(), map(tokens.group, range(1, 3)))

def _compiler(line):
    regex = re.compile(r'compiler:(.*)', re.M | re.L)
    return regex.search(line).group(1).strip()

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

def parse_file(input_, output, mode, mtype, deg, argcmd, listsolution, log):
    with open(input_, 'r') as infile:
        outfile = open(output, mode)
        for line in infile:
            if line.startswith('random seed'):
                seed = _random_seed(line)
            elif line.startswith('input'):
                n, m, k, inCost, inTime, inPeak, inCurr = _input(line)
            elif line.startswith('root build'):
                rZero, rPos, rAdj, rTerm, rTotaltime, rPeak, rCurr = _root_build(line)
            elif line.startswith('erickson'):
                if listsolution:
                    eZero, eKernel, eKernelBw, etraceback, eTotal, eCost, ePeak, eCurr = _erickson_traceback(line)
                else:
                    eZero, eKernel, eKernelBw, eTotal, eCost, ePeak, eCurr = _erickson(line)
                #end if
            elif line.startswith('dijkstra'):
                diZero, diHinsert, diVisit, diTotal, diMemband, diGtotal, diPeak = _dijkstra(line)
            elif line.startswith('gen-unique'):
                gType, gN, gM, gK, gCost, gSeed = _genunique(line)
            elif line.startswith('solution'):
                edgelist = _solution(line)
            elif line.startswith('terminals'):
                 terminals = _terminals(line)
            elif line.startswith('command done.'):
                cmdTime = _command_done(line)
            elif re.match('grand total', line, re.M|re.L):
                gTotaltime, gPeak = _grand_total(line)
            elif line.startswith('host'):
                host = _host(line)
            elif line.startswith('build'):
                threads, heap = _build_type(line)
            elif line.startswith('compiler'):
                compiler = _compiler(line)

        outfile.write("%8d %10d %2d %10d %10d %8.2lf %8.2lf"% 
                       (int(n), int(m), int(k), int(inCost), int(seed), 
                       float(inPeak), float(inCurr)))
        outfile.write(" %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf %7.2lf %7.2lf"%
                        (float(rZero), float(rPos), float(rAdj), float(rTerm),
                        float(rTotaltime), float(rPeak), float(rCurr)))
        if argcmd == 'erickson':
            outfile.write(" %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf"% \
                            (float(eZero), float(eKernel), float(eKernelBw), \
                            float(eTotal), float(eCost), float(ePeak), float(eCurr)))
        elif argcmd == 'dijkstra-apsp':
            outfile.write(" %8.2lf %8.2lf %7.2lf"% \
                            (float(diApsp), float(diTotal), float(diPeak)))
        elif argcmd == 'dijkstra':
            outfile.write(" %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf %8.2lf %7.2lf"% \
                            (float(diZero), float(diHinsert), float(diVisit), \
                            float(diTotal), float(diMemband), float(diGtotal), \
                            float(diPeak)))
            
        outfile.write(" %8.2f %7.2lf"%
                        (float(gTotaltime), float(gPeak)))
        outfile.write(" %d %d %d %d %d %s"% 
                        (int(gN), int(gM), int(gK), int(gCost), int(gSeed),
                        gType))
        outfile.write(" %s"% (compiler))
        outfile.write(" %s %s"% (threads, heap))
        outfile.write('\n')
        outfile.close()

    if(int(inCost) != int(eCost) and int(inCost) != -1):
        _logerr(log, "Error: inCost = %d dwCost = %d\n"%\
                         (int(inCost), int(eCost)))
    if listsolution:
        ret = verify_graph(edgelist, terminals)
        if(ret == 0):
            return
        elif(ret == -1):
            _logerr(log, "fail, graph is disconnected\n")
        else:
            _logerr(log, "fail, %d terminal not covered\n"% (ret))
        #end if
    #end if
#end of parse_file

#***************************************** command line parser ****************
def _add_arg(g, sop, op, nargs_, required_, type_, default_, choices_=[], help_=""):
    if len(choices_):
        g.add_argument( sop, op, nargs=nargs_, required=required_,
            type=type_, default=default_, choices=choices_)
    else:
        g.add_argument( sop, op, nargs=nargs_, required=required_,
            type=type_, default=default_)

def cmd_parser():
    #TODO: too many arguments is a mess. 
    # find a better way if possible
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Functionality')
    g.add_argument('-run', '--run-experiments', action='store_true')
    g.add_argument('-parse', '--parse', action='store_true')

    ## build args
    g = parser.add_argument_group('Build')
    g.add_argument('-b', '--build', action='store_true')
    _add_arg(g, '-bt', '--build-type', '+', False, str,
             ['READER_DEFAULT'],[])

    _add_arg(g, '-m',  '--machine-type', '?', False, str, 'cpu-corei5', 
             ['cpu-corei5', 'cpu-hsw', 'cpu-hsw-largemem', 'cpu-hsw-hugemem'])
    _add_arg(g, '-e','--exe', '?', False, str, 'reader-el', 
             ['reader-el'])
    _add_arg(g, '-rep', '--report-dir', '?', False, str, 'Report')

    ## run arguments
    _add_arg(g, '-arg', '--arg-cmd', '?', False, str,
             'dreyfus', ['dijkstra', 'erickson'])
    g.add_argument('-list', '--list-solution', action='store_true')
    ## graph 
    g = parser.add_argument_group('Graph generator')
    _add_arg(g, '-g',  '--gen', '?', False, str, 'gen-unique', 
             ['gen-unique', 'gen-count', 'gen-natural'])
    _add_arg(g, '-gt', '--graph-type', '?', False, str, 'regular', 
             ['regular', 'powlaw', 'clique', 'dbpedia-en', 'dbpedia-fr', \
              'dbpedia-it', 'dbpedia-pt', 'dbpedia-all', 'web-berkstan'])
    _add_arg(g, '-ft', '--graphfile-type', '?', False, str, 'bin', 
             ['bin', 'ascii'])
    _add_arg(g, '-n', '--nodes', '+', False, int, [1024])
    _add_arg(g, '-d', '--deg', '+', False, int, [20])
    _add_arg(g, '-k', '--terminals', '+', False, int, [10])
    _add_arg(g, '-al', '--alpha', '?', False, float, -0.5)
    _add_arg(g, '-w', '--weight', '?', False, int, 10000)
    _add_arg(g, '-ew', '--edge-weight', '?', False, int, 10000)
    _add_arg(g, '-r', '--repeats', '?', False, int, 1)
    _add_arg(g, '-gr', '--graph-repeats', '?', False, int, 1)
    _add_arg(g, '-vr', '--vertex-repeats', '?', False, int, 1)
    _add_arg(g, '-s', '--seed', '?', False, int, 123456789 ) 

    ## plot args
    g = parser.add_argument_group('Plot')
    g.add_argument('-p', '--plot-graph', action='store_true')
    _add_arg(g, '-gp', '--gnuplot-file', '+', False, str, [])


    ## additional
    g = parser.add_argument_group('Optional')
    g.add_argument('-err', '--keep-error', action='store_true')
    g.add_argument('-bin', '--delete-bin', action='store_true')
    g.add_argument('-tar', '--archive', action='store_true')
    g.add_argument('-imp', '--impromptu-mode', action='store_true')
    g.add_argument('-pr', '--print', action='store_true')
    _add_arg(g, '-log', '--log-mode', '?', False, str, 'w', ['r', 'w', 'a'])

    return parser
#end of cmdParse

#**************************************** extra work *************************
def _logerr(log, msg):
    sys.stderr.write("error: %s"% (msg))
    sys.stderr.flush()
    if __RUN_EXP:
        log.write("error: %s"% (msg))

def _logmsg(log, msg):
    sys.stdout.write("log: %s"% (msg))
    sys.stdout.flush()
    if __RUN_EXP:
        log.write("log: %s"% (msg))

def _call(cmd, log=sys.stdout):
    ret = call(cmd, shell=True)
    if ret != 0:
        if ret < 0:
            _logerr(log, "killed by signal %d\n"% (ret))
            return 0
        else:
            _logerr(log, "%s command failed with return code %d\n"% (cmd, ret))
            return 0
    return 1
    #else:
    #   _logmsg(log, '%s success'% (cmd))

def _sys_details(log):
    log.write('system details: \n\t')
    log.write("date        = %s \n\t"% (time.strftime("%d/%m/%y")))
    log.write("time        = %s \n\t"% (time.strftime("%H:%M:%S")))
    sysname, nodename, release, version, machine = [e for e in list(os.uname())]
    log.write("nodename    = %s\n\t"% (nodename))
    log.write("sysname     = %s\n\t"% (sysname))
    log.write("release     = %s\n\t"% (release))
    log.write("version     = %s\n\t"% (version))
    log.write("machine     = %s\n"% (machine))

def _archive(dir_):
    if os.path.exists(dir_):
        cmd = "tar -zcvf %s.tar.gz %s 1>/dev/null"% (dir_, dir_)
        call(cmd, shell=True)
        cmd = "rm -rf %s"% (dir_)
        call(cmd, shell=True)

def _build(exe_):
    cmd = "make %s "% (exe_)
    return _call(cmd, sys.stdout)
    
def _copy(src, dst, list_):
    for file_ in list_:
        if os.path.exists("%s/%s"% (src, file_)):
            cmd = "cp %s/%s %s/"% (src, file_, dst)
            call(cmd, shell=True)
        else:
            print("file '%s' doesnot exists"% (file_))
            return 0 #fail
    return 1 #success

def _delete(list_):
    for file_ in list_:
        cmd = "rm -rf %s"% (file_)
        call(cmd, shell=True)

def _printlist(name_, file_, mode, list_=[]):
    with open(file_, mode) as outfile:
        outfile.write("\t%s = [ "% (name_))
        for e in list_:
            outfile.write("%d "% (e))
        outfile.write("]\n")

def _printlist(outfile, name_, list_=[]):
    outfile.write("%s = [ "% (name_))
    for e in list_:
        outfile.write("%s "% str(e))
    outfile.write("]\n")

def _handle_dir(dir_, impromptu):
    choice = 'no'

    if impromptu:
        sys.stdout.write('delete contents: [yes/no] ')
        choice = raw_input().lower()
        if choice == 'yes':
            sys.stdout.write('deleting report files...\n')
            cmd = "rm -rf %s/*"% (dir_)
            _call(cmd, sys.stdout)
            return 1
    else:
        sys.stdout.write('deleting report files...\n')
        ## ***no deletion of reports***
        #cmd = "rm -rf %s/*"% (dir_)
        #_call(cmd, sys.stdout)
        return 1
    ## end if impromptu

    if choice == 'yes':
        sys.stdout.write('deleting report files...\n')
        subdirList = os.walk(dir_)
        os.chdir(dir_)
        for subdir in subdirList:
            os.chdir(subdir)
            _delete(glob.glob('*.err'))
            _delete(glob.glob('*.log'))
            _delete(glob.glob('*.report'))
            os.chdir('../')
        _delete(glob.glob('*.report'))
        _delete(glob.glob('*.gp'))
        _delete(glob.glob('*.eps'))
        _delete(glob.glob('*.pdf'))
        cmd = "rm -f %s"% (" ".join(files))
        _call(cmd, sys.stdout)
        os.chdir('../')
    else:
        sys.stdout.write('choose different report directory \n')
        sys.stdout.write('exiting program...\n')
        sys.exit()
    ## end if choice
## end _handle_dir()

# TODO: finalise the column values
def _printcolumnvalues(log):
    log.write('\nColumn values: \n')
    list_ = ['host', 'yes', 'kmotif', 'n', 'm', 'k', 'seed', 't', \
             'inPeak', 'inCurr', 'zero', 'pos', 'adj', 'adjSort', \
             'shade', 'rTotalTime', 'rPeak', 'rCurr', 'sum_', 'oracleTime', \
             'transRate', 'MulRate', 'deg', 'gl', 'inSize', 'instrRate', \
             'cmdTime', 'totalTime', 'gPeak', 'sieveTime', 'genfTime', \
             'outer', 'dg/VERTICES_PER_GENF_THREAD', 'db', 'build']

    log.write("%s  %s\n"% ('col-num', 'col-value'))
    for index in range(0, len(list_)):
        log.write("%7d  %s\n"% (index + 1, list_[index]))

def _get_reportdir(buildtype, mtype, graphtype, plottype, argcmd, deg):
    dir_ = "%s_%s_%s_%s_%s_%d_dir"% \
            (buildtype, mtype, graphtype, plottype, argcmd, deg[0])
    return dir_

def _get_reportfile(buildtype, mtype, graphtype, plottype, argcmd, deg):
    file_ = "%s_%s_%s_%s_%s_%d"% \
            (buildtype, mtype, graphtype, plottype, argcmd, deg[0])
    file_ += '.report'
    return file_

def _get_plottype(nn, kk):
    plottype = ''
    if len(nn) == 1 and len(kk) != 1:
        plottype = 'KvsDT'
    elif len(nn) != 1 and len(kk) == 1:
        plottype = 'MvsDT'
    return plottype

def _get_genCmd(gen, reportDir, graphtype, n, d, k, genSeed, ftype, tmplog,\
                alpha, w, ew):
    genCmd = ''
    if gen == 'gen-unique' and graphtype == 'powlaw':
        genCmd = "%s/%s %s %d %d %f %d %d %d %d -%s 2>>%s"% \
            (reportDir, gen, graphtype, n, d, alpha, w, \
                k, ew, genSeed, ftype, tmplog) 
    elif gen == 'gen-natural':
        m = int((n * d) / 2)
        genCmd = "%s/%s.py -it bin -i %s -s %d -m %d -k %d -ot %s 2>>%s"% \
            (reportDir, gen, genSeed, m, k, ftype, tmplog)
    else:
        genCmd = "%s/%s %s %d %d %d %d %d -%s 2>>%s"% \
            (reportDir, gen, graphtype, n, d, k, ew, genSeed, \
                ftype, tmplog)
    ## end if
    return genCmd
## end _get_genCmd()

def _get_exeCmd(reportDir, exe, ftype, seed, argcmd, errFile, listsolution):
    if listsolution:
        exeCmd = "%s/%s -%s -oracle -seed %d -%s -list 2>>%s"% \
                 (reportDir, exe, ftype, seed, argcmd, errFile)
    else:
        exeCmd = "%s/%s -%s -oracle -seed %d -%s 2>>%s"% \
                 (reportDir, exe, ftype, seed, argcmd, errFile)
    #endif
    return exeCmd
#end _get_exeCmd()
#*************************************** Main function ************************
def main():

    parser = cmd_parser()
    opts = vars(parser.parse_args())

    ## script can do??
    build = opts['build']
    run = opts['run_experiments']
    parse = opts['parse']
    plotgraph = opts['plot_graph']
    printcmd = opts['print']

    ## build input
    mtype = opts['machine_type']
    #multigpu = opts['multigpu']
    buildtypeList = opts['build_type']

    ## reader arg command
    argcmd = opts['arg_cmd']
    listsolution = opts['list_solution']

    ## run input 
    exe = opts['exe']
    gen = opts['gen']
    graphtype = opts['graph_type']
    nn = opts['nodes']
    dd = opts['deg']
    kk = opts['terminals']
    ftype = opts['graphfile_type']
    alpha = opts['alpha']
    weight = opts['weight']
    edgeweight = opts['edge_weight']
    repeats = opts['repeats']
    gcount = opts['graph_repeats']
    initSeed = opts['seed']
    archive = opts['archive']
    keepErr = opts['keep_error']
    deleteBin = opts['delete_bin']
    repDir = opts['report_dir']
    impromptu = opts['impromptu_mode']
    logmode = opts['log_mode']

    ## plot input
    gplotfileList = opts['gnuplot_file']
    plottype = _get_plottype(nn, kk)

    sys.stderr.write('invoked as: ')
    for i in range(0, len(sys.argv)):
        sys.stderr.write(" %s"% (sys.argv[i]))
    sys.stderr.write('\n\n')

    curDir = os.getcwd()
    srcDir = os.path.abspath("../")
    exeDir = "%s/reader"% (srcDir)
    genDir = "%s/graph-gen"% (srcDir)
    repDirName = os.path.join(curDir, repDir)

    if os.path.exists(repDirName):
        sys.stdout.write("directory '%s' exits \n"% (repDirName))
        #os.chdir(repDirName)
        #cmd = 'rm -f *.report'
        #_call(cmd, sys.stdout)
        #os.chdir('../')
    else:
        cmd = "mkdir -p "+repDirName
        _call(cmd, sys.stderr)
    ## end if

    if build:
        # build graph generator
        cmd = "make " + gen
        os.chdir(genDir)
        _call(cmd, sys.stderr)
        os.chdir(curDir)

        # push into report directory
        os.chdir(repDirName)
        for buildtype in buildtypeList:
            #for d in dd:
            makeCmd = "make " + buildtype
            os.chdir(exeDir)
            _call(makeCmd, sys.stderr)
            os.chdir(repDirName)

            reportDir = _get_reportdir(buildtype, mtype, graphtype, plottype, 
                                       argcmd, dd)
            if os.path.exists(reportDir) == False:
                cmd = 'mkdir -p '+reportDir
                _call(cmd, sys.stderr)
            ## end if

            exe = buildtype
            cmd = "cp %s/%s %s"% (exeDir, exe, reportDir)
            _call(cmd, sys.stdout)

            cmd = "cp %s/%s %s"% (genDir, gen, reportDir)
            _call(cmd, sys.stderr)
        ## end for dd
        ## end for buildtype
        os.chdir('../')
    ## end if build

    if run:
        global __RUN_EXP 
        __RUN_EXP = True
        logfile = repDirName+"/run.log"
        log = open(logfile, logmode)
        _sys_details(log)
        log.write('\nbuild options: \n\t')
        log.write("build       = %d\n\t"% (build))
        log.write("machine-type= %s\n\t"% (mtype))
        _printlist(log, "build-type ", (buildtypeList if build else []))
        log.write("\nexe options: \n\texe         = %s \n\t"% (exe))
        log.write("init seed   = %d\n"% initSeed)

        log.write('generator options: \n')
        log.write("\tgen           = %s \n\tgraph-type    = %s \n"% \
                    (gen, graphtype))
        log.write("\trepeats       = %d\n"% (repeats))
        log.write("\tgraph-repeats = %d\n"% (gcount))
        _printlist(log, "\tn", nn)
        _printlist(log, "\td", dd)
        _printlist(log, "\tk", kk)
        log.write("\tw = %d\n"% (weight))
        log.write("\tedge weight = %d\n"% (edgeweight))
        log.write("\talpha = %s\n"% (alpha))

        log.write('plot options: \n')
        log.write("\tplot-graph    = %d\n"% (plotgraph))
        log.write("\tgnuplot-files = %s"% (" ".join(gplotfileList)))
        _printcolumnvalues(log)

        _logmsg(log, 'command start\n')
        _logmsg(log, "date = %s\n"% (time.strftime("%d/%m/%y")))
        _logmsg(log, "time = %s\n"% (time.strftime("%H:%M:%S")))
    elif parse:
        log = sys.stdout
    ## end if run

    # start report generation
    if run or parse:
        # push into report directory
        os.chdir(repDirName)
        for buildtype in buildtypeList:
            #for d in dd:
            exe = buildtype
            random.seed(initSeed)
            reportDir = _get_reportdir(buildtype, mtype, graphtype, plottype,
                                       argcmd, dd)
            if os.path.exists(reportDir):
                # clean up the mess
                os.chdir(reportDir)
                if run:
                    _logmsg(log, "directory '%s' exists, deleting contents...\n"% 
                                 (reportDir))
                    cmd = 'rm -f *.log *.report *.err'
                    _call(cmd, sys.stderr)
                ## end if
                os.chdir('../')
            else:
                sys.stderr.write("directory '%s' doesnot exist...\n"% 
                                 (reportDir))
                sys.exit();
            ## end if
            
            if parse:
                reportfile = _get_reportfile(buildtype, mtype, graphtype, 
                                             plottype, argcmd, dd)
                report = os.path.join(reportDir, reportfile)

                if os.path.exists(report):
                    _logmsg(log, "file '%s' exist, deleting file...\n"% (report))
                    _delete([report])
                ## end if
            ## end if parse
                
            #_logmsg(log, "%s: "% (buildtype))
            if os.path.exists("%s/%s"% (reportDir, buildtype)) == False:
                _logerr(log, "file '%s' doesnot exist\n"% (buildtype))
                sys.exit()
            ## end if
            if os.path.exists("%s/%s"% (reportDir, gen)) == False:
                _logerr(log, "file '%s' doesnot exist\n"% (gen))
                sys.exit()
            ## end if
                
                
            _logmsg(log, 'starting run...\n')
            for n, d, k in itertools.product(nn, dd, kk):
                # graph generator using same seed for all repeats
                for g, r in itertools.product(range(0, gcount), range(0, repeats)):
                    if run:
                        log.write("\t[ %10d %6d %4d %4d %4d ] "% (n, d, k, g, r))
                    sys.stdout.write("\t[ %10d %6d %4d %4d %4d ] "% (n, d, k, g, r))
                    sys.stdout.flush()
                    tstart = time.time()

                    if r == 0:
                        genSeed = random.randint(512, 2<<32)
                    #print( n, d, k, g, r)
                    tmplog   = "%s/n%d-d%d-k%d-g%d-r%d.log"% \
                                (reportDir, n, d, k, g, r)
                    errFile  = "%s/n%d-d%d-k%d-g%d-r%d.err"% \
                                (reportDir, n, d, k, g, r)
                    # prepare command 
                    genCmd = _get_genCmd(gen, reportDir, graphtype, n, d, k, \
                                         genSeed, ftype, tmplog, alpha, weight,\
                                         edgeweight)
                    # new seed for every run of reader
                    seed = random.randint(512, 2<<32)
                    exeCmd = _get_exeCmd(reportDir, exe, ftype, seed, argcmd,
                                        errFile, listsolution)
                    #exeCmd = "%s/%s -%s -oracle -seed %d -%s 2>>%s"% \
                    #         (reportDir, exe, ftype, seed, argcmd, errFile)
                    # run
                    cmd = "%s | %s >> %s"% (genCmd, exeCmd, tmplog)

                    if printcmd:
                        sys.stdout.write(cmd)
                    
                    if run:
                        _call(cmd, log)
                    
                    if os.path.exists(errFile) and os.stat(errFile).st_size > 0 :
                        if 'error' in open(errFile).read():
                            _logerr(log, \
                                "n = %d d = %d k = %d n%d-d%d-k%d-g%d-r%d.err\n"% \
                                (n, d, k, n, d, k, g, r))
                        if 'Assert' in open(errFile).read():
                            _logerr(log, \
                                "n = %d d = %d k = %d n%d-d%d-k%d-g%d-r%d.err\n"% \
                                (n, d, k, n, d, k, g, r))
                    else:
                        if parse:
                            try:
                                parse_file( tmplog, report, "a", mtype, d,
                                            argcmd, listsolution, log)
                            except:
                                sys.stdout.write("Error: parsing failed %s\n"% 
                                    (traceback.format_exc()))
                                sys.exit()
                        ## Remove error file, if no error
                        if keepErr == False:
                           _delete([errFile])
                    ## end if errFile
                    if run:
                        log.write(" %10.5f s\n"% (time.time() - tstart))
                    sys.stdout.write(" %10.5f s\n"% (time.time() - tstart))
                    sys.stdout.flush()
                    ## end for g, r
                ## end for n,d,k
            if deleteBin:
                _delete(["%s/%s"% (reportDir, exe), "%s/%s"% (reportDir, gen)])
            ## end for dd
        ## end for buildtypeList
        os.chdir('../')
    ## end if parse or run

    if run:
        _logmsg(log, 'run finished\n')
        _logmsg(log, "date = %s\n"% (time.strftime("%d/%m/%y")))
        _logmsg(log, "time = %s\n"% (time.strftime("%H:%M:%S")))
        _logmsg(log, '-------------------------------------------------------\n')
        log.close()
    ## end if run

    ## plot graph
    delEps = True
    if plotgraph:
        sys.stdout.write('plotting graphs..\n')
        plotDir = 'plots'
        _copy(plotDir, repDirName, gplotfileList)
        ## get report files
        os.chdir(repDirName)
        reportfileList = glob.glob(os.path.join('*','*.report'))
        _copy('.', '.', reportfileList)
        #cmd = 'find ./ -type f | grep -i .report$ | xargs -i cp {} .'
        #_call(cmd, sys.stderr)

        ## plotting to eps
        plotfileList = glob.glob("*.gp")
        for plotfile in plotfileList:
            sys.stdout.write("plotting %s\n"% (plotfile))
            cmd = "gnuplot %s"% (plotfile)
            call(cmd, shell=True)
        ## convert eps to pdf
        epsfileList = glob.glob("*.eps")
        for epsfile in epsfileList:
            sys.stdout.write("converting '%s' to pdf\n"% (epsfile))
            cmd = "epspdf %s"% (epsfile)
            call(cmd, shell=True)
        if delEps:
            cmd = "rm -f %s"% (" ".join(epsfileList))
            call(cmd, shell=True)
        sys.stdout.write('plotting completed\n')
        os.chdir('../')
    ## end if plot
    if archive:
        _archive(repDirName)
## end of main

if __name__ == "__main__":
    main()
