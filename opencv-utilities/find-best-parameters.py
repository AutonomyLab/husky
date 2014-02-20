#! /usr/bin/env python



import os, sys


filename = sys.argv[1]

with open(filename) as f:
    content = f.readlines()

    best = -sys.maxint
    best_string = None

    for string in content:
        toks = string.split(" ")
        err = toks[4]
        errnum = err.split("=")[1].split(",")[0]

        clus = toks[5]
        cnum = clus.split("=")[1].split(",")[0]

        errnum = float(errnum)
        cnum = float(cnum) + 1

        score = errnum / cnum

        if score > best:
            best = score
            best_string = string
            print "Best so far:\n%s" % best_string

    print best_string
