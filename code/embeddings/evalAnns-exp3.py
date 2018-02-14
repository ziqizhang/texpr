# -*- coding: utf-8 -*-

# Script for calculating various evaluation measures from the
# results returned by the annotators and the original file from which the file for the
# annotators was created.
# This will make same sanity checks to make sure the files correspond to each other.

# Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
import scipy.stats
import statistics

if len(sys.argv) != 3:
    print("ERROR: need the following arguments: originalfile annotatorfile", file=sys.stderr)
    sys.exit(1)

debug = False
verbose = True

origFile = sys.argv[1]
annFile = sys.argv[2]

# first read in the classinfo file and create the data structures we need
# for sampling classes and keywords
n_rows_orig = 0

# read in the original file and store everything in a list of tupes
origData = []
lastRownr = -1
with open(origFile) as infile:
    for line in infile:
        line = line.strip()
        # (rownr,classname,cwordOrKw,kword,measurename,rank,sim,whatitis) = line.split("\t")
        fields = line.split("\t")
        (rownr, classname, cwordOrKw, kword, measurename, rank, sim, whatitis) = fields
        # print("DEBUG: orig fields, rownr=%s,classname=%s,cwordOrKw=%s,kw=%s,measname=%s,rank=%s,sim=%s,whatisit=%s" %
        #      (rownr, classname, cwordOrKw,  kword, measurename, rank, sim, whatitis))
        rownr = int(rownr)
        if rownr != (lastRownr+1):
            print("ERROR: in orig file, expected row nr", lastRownr+1, "but got", rownr, file=sys.stderr)
            sys.exit(1)
        lastRownr += 1
        origData.append(fields)
        n_rows_orig += 1

# read in the annotator's file
lastRownr = -1
n_rows_ann = 0
annData = []
with open(annFile) as infile:
    for line in infile:
        n_rows_ann += 1
        line = line.strip()
        fields = line.split("\t")
        if len(fields) >= 4:
            (rownr, classname, cwordOrKw, annotation) = fields[0:4]
        else:
            print("ERROR, now this is odd, we expect 5 or 4 columns but got ", len(fields), "in row", n_rows_ann, file=sys.stderr)
            sys.exit(1)
        rownr = int(rownr)
        if rownr != (lastRownr+1):
            print("ERROR: in ann file, expected row nr", lastRownr+1, "but got", rownr, file=sys.stderr)
            sys.exit(1)
        lastRownr += 1
        annData.append(fields[0:4])

if n_rows_orig < n_rows_ann:
    n_rows_orig = n_rows_ann  # we only remove the rows after the orig gets created, so this should be ok
    origData = origData[0:n_rows_ann]
elif n_rows_orig < n_rows_ann:
    print("ERROR: orig file has rows", n_rows_orig, "ann file has rows", n_rows_ann, file=sys.stderr)
    sys.exit(1)

simData = []
simidfData = []
for i in range(len(annData)):
    (rownr1, classname1, cwordOrKw1, kword, measurename, rank, sim, whatitis) = origData[i]
    if len(annData[i]) != 4:
        print("ERROR: odd, not 4 columns of annData in row", i, "but", len(annData[i]), "data is", annData[i])
    (rownr2, classname2, cwordOrKw2, annotation) = annData[i]
    annotation = int(annotation)
    rank = int(rank)
    if rownr1 != rownr2:
        raise Exception("ERROR")
    if classname1 != classname2:
        print("Error class names differ: %s/%s" % (classname1, classname2), file=sys.stderr)
    if cwordOrKw1 != cwordOrKw2:
        print("ERROR: keywords from orig/ann differ: %s/%s" % (cwordOrKw1, cwordOrKw2), file=sys.stderr)
    if measurename == "simonly":
        simData.append((rank,annotation))
    else:
        simidfData.append((rank,annotation))

print("Got sim data entries:", len(simData))
print("Got simidf data entries:", len(simidfData))


# calculate stats for the sim scores:
print("Original corpusword scores - N:", len(simData))
simDataRanks = [x for x, y in simData]
simDataScores = [y for x, y in simData]
n = len(simData)*1.0
print("Original corpusword scores - mean:", statistics.mean(simDataScores))
print("Original corpusword scores - median:", statistics.median(simDataScores))
print("Original corpusword scores - mode:", statistics.mode(simDataScores))
print("Original corpusword scores - counts 0/1/2:", simDataScores.count(0), "/", simDataScores.count(1), "/", simDataScores.count(2))
print("Original corpusword scores - relative 0/1/2:", simDataScores.count(0)/n, "/", simDataScores.count(1)/n, "/", simDataScores.count(2)/n)
print("Original corpusword scores - relative 0/1+2:", simDataScores.count(0)/n, "/", (simDataScores.count(1)+simDataScores.count(2))/n)
print("Original corpusword scores - Spearman rank correlation:", scipy.stats.spearmanr(simDataRanks, simDataScores))
print("Original corpusword scores - Kendall tau:", scipy.stats.kendalltau(simDataRanks, simDataScores))
print("Original corpusword scores - values:", sorted(simDataScores))


# calculate stats for the simidf scores:
print("Textrank corpusword scores - N:", len(simidfData))
n=len(simidfData)*1.0
simidfDataRanks = [x for x, y in simidfData]
simidfDataScores = [y for x, y in simidfData]
print("IDF corpusword scores - mean:", statistics.mean(simidfDataScores))
print("IDF corpusword scores - median:", statistics.median(simidfDataScores))
print("IDF corpusword scores - mode:", statistics.mode(simidfDataScores))
print("IDF corpusword scores - counts 0/1/2:", simidfDataScores.count(0), "/", simidfDataScores.count(1), "/", simidfDataScores.count(2))
print("Original corpusword scores - relative 0/1/2:", simidfDataScores.count(0)/n, "/", simidfDataScores.count(1)/n, "/", simidfDataScores.count(2)/n)
print("Original corpusword scores - relative 0/1+2:", simidfDataScores.count(0)/n, "/", (simidfDataScores.count(1)+simidfDataScores.count(2))/n)
print("IDF corpusword scores - Spearman rank correlation:", scipy.stats.spearmanr(simidfDataRanks, simidfDataScores))
print("IDF corpusword scores - Kendall tau:", scipy.stats.kendalltau(simidfDataRanks, simidfDataScores))
print("IDF corpusword scores - values:", sorted(simidfDataScores))
