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
        line = line.strip()
        fields = line.split("\t")
        if len(fields) == 5:
            (rownr,classname,cwordOrKw,annotation,comment) = fields
        else:
            (rownr, clasname, cwordOrKw, annotation) = fields
            fields.append("")
        rownr = int(rownr)
        if rownr != (lastRownr+1):
            print("ERROR: in ann file, expected row nr",lastRownr+1,"but got",rownr,file=sys.stderr)
            sys.exit(1)
        lastRownr += 1
        annData.append(fields)
        n_rows_ann += 1

if n_rows_orig != n_rows_ann:
    print("ERROR: orig file has rows", n_rows_orig, "ann file has rows", n_rows_ann, file=sys.stderr)
    sys.exit(1)

# merge the columns and also split up into the three sets of rankings/ranking pairs we need
kwData = []
simData = []
textrankData = []
for i in range(len(annData)):
    (rownr1, classname1, cwordOrKw1, kword, measurename, rank, sim, whatitis) = origData[i]
    (rownr2, classname2, cwordOrKw2, annotation, comment) = annData[i]
    annotation = int(annotation)
    rank = int(rank)
    if rownr1 != rownr2:
        raise Exception("ERROR")
    if classname1 != classname2:
        raise Exception("Error")
    if cwordOrKw1 != cwordOrKw2:
        raise Exception("ERROR")
    if whatitis == "original-kw":
        kwData.append(annotation)
    else:
        if measurename.endswith("-textrank"):
            textrankData.append((rank,annotation))
        else:
            simData.append((rank,annotation))

print("Got kw data entries:", len(kwData))
print("Got sim data entries:", len(simData))
print("Got tr data entries:", len(textrankData))

# calculate stats for the kw scores:
print("Original keyword scores - N:", len(kwData))
print("Original keyword scores - mean:", statistics.mean(kwData))
print("Original keyword scores - median:", statistics.median(kwData))
print("Original keyword scores - mode:", statistics.mode(kwData))

# calculate stats for the sim scores:
print("Original corpusword scores - N:", len(simData))
simDataRanks = [x for x, y in simData]
simDataScores = [y for x, y in simData]
print(sorted(simDataScores))
print("Original corpusword scores - mean:", statistics.mean(simDataScores))
print("Original corpusword scores - median:", statistics.median(simDataScores))
print("Original corpusword scores - mode:", statistics.mode(simDataScores))
print("Original corpusword scores - counts 0/1/2:", simDataScores.count(0), "/", simDataScores.count(1), "/", simDataScores.count(2))
print("Original corpusword scores - Spearman rank correlation:", scipy.stats.spearmanr(simDataRanks, simDataScores))
print("Original corpusword scores - Kendall tau:", scipy.stats.kendalltau(simDataRanks, simDataScores))


# calculate stats for the textrank scores:
print("Textrank corpusword scores - N:", len(textrankData))
textrankDataRanks = [x for x, y in textrankData]
textrankDataScores = [y for x, y in textrankData]
print(sorted(textrankDataScores))
print("Textrank corpusword scores - mean:", statistics.mean(textrankDataScores))
print("Textrank corpusword scores - median:", statistics.median(textrankDataScores))
print("Textrank corpusword scores - mode:", statistics.mode(textrankDataScores))
print("Textrank corpusword scores - counts 0/1/2:", textrankDataScores.count(0), "/", textrankDataScores.count(1), "/", textrankDataScores.count(2))
print("Textrank corpusword scores - Spearman rank correlation:", scipy.stats.spearmanr(textrankDataRanks, textrankDataScores))
print("Textrank corpusword scores - Kendall tau:", scipy.stats.kendalltau(textrankDataRanks, textrankDataScores))
