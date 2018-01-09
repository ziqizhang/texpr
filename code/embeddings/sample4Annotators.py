## Script for creating the final ranking file from the following inputs:
## - the mostSim4Onto-EMB.tsv file (with all the similarities and full ranks)
## - the corpusword-textrank tsv file which maps corpus words to textrank

## the script writes a tsv file which contains the following information:
## - the keyword
## - the corpus word
## - a similarity score name
## - a score
## - a rank
## Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
import os
from embeddingsutils import EmbeddingsUtils
from tqdm import tqdm
import heapq
import numpy as np
from difflib import SequenceMatcher
from collections import OrderedDict

if len(sys.argv) != 4:
    print("ERROR: need the following arguments: mostSim-file, textrank-file, output-file",file=sys.stderr)
    sys.exit(1)

debug=False
verbose=True
mixedCase=True
k = 100  # how many ranks to output

mostSimFile = sys.argv[1]
textRankFile = sys.argv[2]
outputFile = sys.argv[3]

written = 0  # number of rows written, total
notr = 0     # number of times we did not find a textrank for the corpus word

## first read in all the corpus word to text rank mappings and remember them
textRank = {}
n_textranks = 0
print("Reading textranks ...",file=sys.stderr)
with open(textRankFile) as infile:
    for line in infile:
        n_textranks += 1
        line = line.strip()
        (word,tr) = line.split("\t")
        textRank[word] = float(tr)
print("Textrank entries read: ",n_textranks,file=sys.stderr)

## Read in the rows for one keyword-corpusword pair and similarity measure,
## calculate all the additional similarities and then put into a sorting
## queue. Output original similarity plus rank and then the derived
## similarity(ies) plus rank

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
print("processing ...",file=sys.stderr)
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
