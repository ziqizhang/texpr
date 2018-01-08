## Script for creating the final ranking file from the following inputs:
## - the mostSim4Onto-EMB.tsv file (with all the similarities and full ranks)
## - the corpusword-textrank tsv file which maps corpus words to textrank
## - the keyword-class uri tsv file which maps keywords to class URIs and sections

## the script writes a tsv file which contains the following information:
## - the class URL
## - the section
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

if len(sys.argv) != 5:
    print("ERROR: need the following arguments: mostSim-file, textrank-file, classinfo-file, output-file",file=sys.stderr)
    sys.exit(1)

debug=False
verbose=True
mixedCase=True

mostSimFile = sys.argv[1]
textRankFile = sys.argv[2]
classInfoFile = sys.argv[3]
outputFile = sys.argv[4]

## first read in all the corpus word to text rank mappings and remember them
textRank = {}
n_textranks = 0
with open(textRankFile) as infile:
    for line in infile:
        n_textranks += 1
        line = line.strip()
        (word,tr) = line.split("\t")
        textRank[word] = tr

## read in all the keyword to class uri and to section mappings and remember
keyword2info = {}
n_kwinfos = 0
with open(classInfoFile) as infile:
    for line in infile:
        n_kwinfos += 1
        line = line.strip()
        (kw,uri,sec) = line.split("\t")
        keyword2info[kw] = (uri,sec)


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
## 10 - similarity times average key word idf (does not influence ranking since same keyword)
with open(outputFile,"w") as outfile:
    with open(mostSimFile) as infile:
        oldkey = ""   ## key is sim+keyword
        for line in infile:
            line = line.strip()
            (simname,cword,kword,rank,score,unused1,unused2,idfc,idfk,simidfk) = line.split("\t")
            key = simname+"|"+kword
            if key == oldkey:
                pass
            else:
                if oldkey:
                    ## finish and output the previous list
                oldkey = key
                # start a new queue
    # finished the whole input file, but we still have to finish the last key
    # TODO!!
