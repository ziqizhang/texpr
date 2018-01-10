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
    print("ERROR: need the following arguments: mostSim-file, classinfo-file, section, N_c, N_k, N_w",file=sys.stderr)
    sys.exit(1)

debug=False
verbose=True
mixedCase=True

mostSimFile = sys.argv[1]
classinfoFile = sys.argv[2]
ontoSection = sys.argv[3]
N_c = int(sys.argv[4])
N_k = int(sys.argv[5])
N_w = int(sys.argv[6])

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
np.random.seed(1)

print("Reading classinfo ...",file=sys.stderr)
with open(classinfoFile) as infile:
    for line in infile:
        n_classinfo += 1
        line = line.strip()
        (keyword,uri,sec) = line.split("\t")
        if sec != ontoSection:
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

sys.exit(0)

## now go through the mostsim file and load all the corpus word ranked lists
## for all the similarity measures for each of the keywords we have sampled.
## Once we have all these lists, pick one of those lists at random
## Find the length of the list, then use our biased sampling method to
## pick corpus words from that list

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
n_input = 0
print("Processing ...",file=sys.stderr)
with open(outputFile,"w") as outfile:
    with open(mostSimFile) as infile:
        oldkey = ""   ## key is sim+keyword
        for line in infile:
            n_input += 1
            line = line.strip()
            (simname,cword,kword,rank,score,unused1,unused2,idfc,idfk) = line.split("\t")
            ## NOTE: we used a cleaned version of the keyword, I think for creating mostsim4Onto,
            ## so clean the keyword here as well

            key = simname+"|"+kword
            keyword = kword
            if key != oldkey:
                if oldkey:   # oldkey is blank for the first time we get here, nothing yet then
                    ## finish and output the previous list
                    l1 = heapq.nlargest(k,h_tr)
                    r = 0
                    for s,cw in l1:
                        print(scorename+"-textrank",cw,keyword,r,s,sep="\t",file=outfile)
                        written = written + 1
                        r = r + 1
                # now start new heaps for the derived scores
                h_tr = [] # original score times textrank
                scorename = simname
                oldkey = key
                keyword = kword
            # another row for the same key, just calculate the additional sim
            # and add to the heap
            score = float(score)
            tr = textRank.get(cword)
            if tr:
                scoretr = score * tr
                heapq.heappush(h_tr,(scoretr,cword))
            else:
                notr += 1
            # also, write the current score if the rank is < k
            rank = int(rank)
            if rank < k:
                print(scorename,cword,kword,rank,score,sep="\t",file=outfile)
                written = written + 1
    # finished the whole input file, but we still have to finish the last key
    # TODO!!
    l1 = heapq.nlargest(k,h_tr)
    r = 0
    for s,cw in l1:
        print(scorename+"-textrank",cw,keyword,r,s,sep="\t",file=outfile)
        written = written + 1
        r = r + 1

print("Total number of rows written:",written,file=sys.stderr)
print("Number of times textrank lookup failed:",notr,file=sys.stderr)
