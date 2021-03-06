## Script for creating the examples to be evaluated by the annotators.
## This takes the following files as input:
## - the final-EMB.tsv file with all the final rankings for all the similarities
## - the classinfo file which contains the keyword, class uri and section corpusword-textrank
##   (the keywords in this file are not converted yet and will get converted/cleaned in here)
## In addition the script needs to know:
## - which ontology section to choose from
## - number of classes to choose N_c
## - number of keywords to choose per class N_k
## - number of corpus words to choose per keyword N_w

## the script writes a tsv file to stdout which contains the following information:
## - class uri
## - keyword
## - corpus word
## - score that was chosen before the rank
## - rank that was used when sampling that corpus word

## The output file will contain N_c * N_k * N_w entries.

## Good values may be:
## N_c = 20
## N_k = 2
## N_w = 20
## which would yield 800 pairs to evaluate

## Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
import os
from embeddingsutils import EmbeddingsUtils
from tqdm import tqdm
import heapq
import re
import numpy as np
from difflib import SequenceMatcher
from collections import OrderedDict

if len(sys.argv) != 7:
    print("ERROR: need the following arguments: finalSim-file, classinfo-file, section, N_c, N_k, N_w",file=sys.stderr)
    sys.exit(1)

debug=False
verbose=True
mixedCase=True

finalSimFile = sys.argv[1]
classinfoFile = sys.argv[2]
ontoSection = sys.argv[3]
N_c = int(sys.argv[4])
N_k = int(sys.argv[5])
N_w = int(sys.argv[6])

MAXRANK = 100
SEED = 1
SIMNAMES = ["simonly","simidfonly","simidfstr","simonly-textrank","simidfonly-textrank","simidfstr-textrank"]
S = 1.0

written = 0  # number of rows written, total

## first read in the classinfo file and create the data structures we need
## for sampling classes and keywords
n_classinfo = 0
n_keywords = 0
n_classes = 0
n_classinfo_skipped = 0
n_classinfo_taken = 0
uris = set()
kw_uris = set()
kws = set()

# make the random selections repeatable
np.random.seed(SEED)

## for debugging, we read in the final most sim file already here one time
## so we know which keywords occur in it. This allows to sample from just
## the keywords of a cut-down, small size final file
if debug:
    print("DEBUG: reading in final file to find the known kws...",file=sys.stderr)
    known_kw_set = set()
    with open(finalSimFile) as infile:
            for line in infile:
                line = line.strip()
                (simname,cword,kword,rank,score) = line.split("\t")
                known_kw_set.add(kword)
    print("DEBUG: found keywords:",len(known_kw_set),file=sys.stderr)

print("Reading classinfo ...",file=sys.stderr)
with open(classinfoFile) as infile:
    for line in infile:
        n_classinfo += 1
        line = line.strip()
        (keyword,uri,sec) = line.split("\t")
        if sec != ontoSection:
            n_classinfo_skipped += 1
            continue
        if debug and keyword not in known_kw_set:
            n_classinfo_skipped += 1
            continue
        n_classinfo_taken += 1
        keyword = re.sub(r"[—-]"," ",keyword)
        uris.add(uri)
        kws.add(keyword)
        kw_uris.add((keyword,uri))

print("Selected for section:",ontoSection,file=sys.stderr)
print("Classes found:",len(uris),file=sys.stderr)
print("Keywords found:",len(kws),file=sys.stderr)
print("Kw/Classes pairs:",len(kw_uris),file=sys.stderr)
# convert the sets to lists and sort so we can have repeatable sampling results
uris = sorted([u for u in uris])
kw_uris = sorted([ku for ku in kw_uris])
# now first create a list of N_c randomly selected classes
# first get the indices using numpy's method
idxs = np.random.choice(len(uris),N_c,replace=False)
# get the list of classes corresponding to the indices
sampled_uris = [uris[i] for i in idxs]
print("DEBUG: selected classes=",sampled_uris,file=sys.stderr)

## now for each class, first get the list of keywords for that class,
## then sample N_k from that list, add the pair (uri,kw) to a result list
## NOTE/IMPORTANT: we do no add a uri/keyword pair if the keyword was already
## sampled for a different class. In theory this could lead to us not finding
## any keyword for a class, so we check that condition and throw an error if
## that really should happen
## Since we now are sure that there is one class for each kw, we can create
## a mapping from keyword to uri
sampled_uri_kw = []
sampled_kw = set()
keyword2uri = {}
for uri in sampled_uris:
    kwds = [(u,k) for (k,u) in kw_uris if u == uri and k not in sampled_kw]
    if not kwds:
        raise RuntimeError("No keywords found for uri="+uri)
    print("DEBUG: got kwds for uri=",uri,"n=",len(kwds),file=sys.stderr)
    if len(kwds) < N_k:
        print("WARNING: fewer keywords than asked for uri=",uri,"n=",len(kwds),file=sys.stderr)
        n_sample = len(kwds)
    else:
        n_sample = N_k
    idxs = np.random.choice(len(kwds),n_sample,replace=False)
    sampled_kwds = sorted([kwds[i] for i in idxs])
    for (u,k) in sampled_kwds:
        keyword2uri[k] = u
        sampled_kw.add(k)
    sampled_uri_kw.extend(sampled_kwds)
sampled_kw = sorted([k for k in sampled_kw])
print("DEBUG: selected kwds=",sampled_kw,file=sys.stderr)
print("DEBUG: selected uri/kwds=",sampled_uri_kw,file=sys.stderr)

## create a set for the keywords so we can check faster if a keyword is of interest later
kw_set = set()
for kw in sampled_kw:
    kw_set.add(kw)
print("DEBUG: kw_set=",kw_set,file=sys.stderr)
## now go through the final mostsim file and load all the corpus word ranked lists
## for all the similarity measures for each of the keywords we have sampled.
## Once we have all these lists, sample from all the lists in the following way
## until we have N_w corpuswords:
##   for rank number r=1 too maxrank:
##      for each similarity measure list:
##         sample word at rank r from that list with probability 1/r^s

## The columns of the input file should be:
## 1 - name of similarity score: simonly, simidfonly, simidfstr
## 2 - corpus word
## 3 - key word
## 4 - rank according to the similarity score
## 5 - similarity score
## 6 - corpus word as used
## 7 - key words as used
## 8 - idf of corpus word
## 9 - average idf of key words used

## First, lets get all the lists for each of the keywords
## We store the lists in the following way:
## each combination of keyword, listid, and rank is mapped to
## corpusword and score in the map cw4kw
cw4kw = {}
n_input = 0
n_found = 0
n_not_found = 0
found_kw_set = set()
print("Processing ...",file=sys.stderr)
with open(finalSimFile) as infile:
        for line in infile:
            n_input += 1
            line = line.strip()
            (simname,cword,kword,rank,score) = line.split("\t")
            if kword in kw_set:
                found_kw_set.add(kword)
                # print("STORING ",(kword,simname,rank),file=sys.stderr)
                cw4kw[(kword,simname,int(rank))]=(cword,float(score))
                n_found += 1
            else:
                n_not_found += 1
print("DEBUG: ('adjuvants', 'simidfstr', 88) info:",cw4kw.get(('adjuvants', 'simidfstr', 88)),file=sys.stderr)
print("DEBUG: ('adjuvant', 'simonly', 0) info:",cw4kw.get(('adjuvants', 'simonly', 0)),file=sys.stderr)
print("Total number of rows read:",n_input,file=sys.stderr)
print("Total number of lines where kw found:",n_found,file=sys.stderr)
print("Total number of lines where kw NOT found:",n_not_found,file=sys.stderr)
print("Number of kwords found:",len(found_kw_set),"expected:",len(kw_set),file=sys.stderr)

if len(found_kw_set) < len(kw_set):
    raise Error("Not all keywords found!")

## now perform the actual sampling
for (uri,kw) in sampled_uri_kw:
    # collect the corpus word tuples for this kw in this list
    cw_set = set() # to check if we already have that word
    cw_list = []   # the list of cw tuples we sampled, with at most N_w elements
    # we have to repeat the whole sampling until we have enough
    for iteration in range(10000):
        for rank in range(MAXRANK):
            th = 1.0/(float(rank+1)**S)   # the threshold: if random number is smaller, pick word
            for simname in SIMNAMES:
                ## sample the corpus word with probability 1/r^s
                rnd = np.random.random()
                if rnd <= th:
                    # get the info we have stored for this keyword
                    info = cw4kw.get((kw,simname,rank))
                    if not info:
                        print("WARNING: no kw info found for ",(kw,simname,rank),file=sys.stderr)
                        # break out of loop that goes over all the similarity measures
                        break
                    else:
                        (cw,score) = info
                    if cw not in cw_set:
                        cw_set.add(cw)
                        cw_list.append((cw,score,rank,simname))
                if len(cw_list) == N_w:
                    break
            if len(cw_list) == N_w:
                break
        if len(cw_list) == N_w:
            break
    # we now should have at most N_w sampled corpus words for the keyword
    # we can now output the whole bunch
    for (cw,score,rank,simname) in cw_list:
        print(uri,kw,cw,simname,rank,score,sep="\t")
