# -*- coding: utf-8 -*-

# Script for calculating IAA measures from multiple annotator files.
# This assumes that the rows between the annotator files are properly aligned.
# Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
from nltk.metrics.agreement import AnnotationTask
import re
import os.path

if len(sys.argv) < 3:
    print("ERROR: need at least two file name arguments", file=sys.stderr)
    sys.exit(1)

debug = False
verbose = True

inFiles = sys.argv[1:]
matcher = re.compile(r"[0-9]+")
# first we create for each input file a list of tuples (rownumber, classification)
# making sure the classification is an integer from 0-3
tuples = []
tuples2 = []
n_rows_ann = 0
annotatornames = []
for inFile in inFiles:
    with open(inFile, "rt") as infile:
        print("DEBUG: file is", inFile, file=sys.stderr)
        m = re.search(r'(_|-)([a-zA-Z0-9]+)_?\.tsv', inFile)
        if not m:
            raise Exception("File name does not match: %s" % inFile)
        name = m.group(2)
        annotatornames.append(name)
        for line in infile:
            n_rows_ann += 1
            line = line.strip()
            fields = line.split("\t")
            if len(fields) >= 4:
                (rownr, classname, cwordOrKw, annotation) = fields[0:4]
            else:
                print("ERROR, now this is odd, we expect 5 or 4 columns but got ", len(fields), "in row", n_rows_ann,
                      "in file", inFile,
                      file=sys.stderr)
                sys.exit(1)
            tuples.append((inFile, rownr, annotation))
            if annotation == "1" or annotation == "2":
                annotation2 = "1"
            elif annotation == "0":
                annotation2 = "0"
            else:
                raise Exception("Annotation is not 0,1,2 but %r" % annotation)
            tuples2.append((inFile, rownr, annotation2))
            # print("DEBUG: adding (%s, %s, %s)" % (inFile, rownr, annotation), file=sys.stderr)

print("IAA for files:")
for file in inFiles:
    print(" ", os.path.basename(file))
print("Number of annotators:", len(inFiles))
print("Annotators:", annotatornames)

task = AnnotationTask(data=tuples)
print()
print("IAA: 3 classes, all different:")
print("Cohen kappa (avg): ", task.kappa())
print("Fleiss kappa: ", task.multi_kappa())
print("Krippendorf alpha: ", task.alpha())
print("Scott pi: ", task.pi())


def dist(a, b):
    ia = int(a)
    ib = int(b)
    return abs(ia-ib)/2.0


task2 = AnnotationTask(data=tuples,distance=dist)
print()
print("IAA: 3 classes, distence based:")
print("Cohen kappa (avg): ", task2.kappa())
print("Fleiss kappa: ", task2.multi_kappa())
print("Krippendorf alpha: ", task2.alpha())
print("Scott pi: ", task2.pi())

task3 = AnnotationTask(data=tuples2)
print()
print("IAA: 2 classes")
print("Cohen kappa (avg): ", task3.kappa())
print("Fleiss kappa: ", task3.multi_kappa())
print("Krippendorf alpha: ", task3.alpha())
print("Scott pi: ", task3.pi())