## simple script which reads in two tsv files:
## * lst file of ontology-related keywords/phrases
## * tsv file of (word,idf) pairs from the corpus
## and outputs to stdout a tsv file with the following fields
## * keyword
## * corpus word
## * similarity
## This is limited to those keyword/corpus word pairs where
## for each keyword for a class, the word had highest similarity
## This means the program will output as many lines as there are words in the second
## file: for each corpus word, the keyword which matches best and its similarity

## output file format is tsv with the following columns:
## * score name, one of simonly, simidfonly, simidfstr
## * corpus word
## * keyword
## * rank number (0 is highest, 1 is next etc.)
## * score value

## Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
import os
from embeddingsutils import EmbeddingsUtils
from tqdm import tqdm
import heapq
import numpy as np
from difflib import SequenceMatcher

if len(sys.argv) != 5:
    print("ERROR: need the following arguments: keywords lst file, corpus words tsv file, embeddings path, 0/1 (0=lowercase/1=mixed case embeddings)",file=sys.stderr)
    sys.exit(1)

debug=False
verbose=True

keywordsFile = sys.argv[1]
corpusWordsFile = sys.argv[2]
embFile = sys.argv[3]
mixedCase = (sys.argv[4]=="1")
print("INFO: using mixed case settings for the embeddings!",file=sys.stderr)
eu = EmbeddingsUtils()
eu.setIsCaseSensitive(mixedCase)
eu.setFallBackToLower(mixedCase)
eu.setFilterStopwords(True)
eu.setDebug(False)
eu.setVerbose(verbose)
eu.loadEmbeddings(embFile)

MIN_EMB_SIM=0.40      # ignore anything where the embedding similarity is less
KEEP_FRACTION=0.2    # keep that portion of highest scoring words, after filtering

keywords = set()
print("Reading ontology keywords ...",file=sys.stderr)
with open(keywordsFile) as inp:
    for line in inp:
        (keyword,cname,other) = line.split("\t")
        ## cname = cname[4:]
        keywords.add(keyword)

print("INFO: got keywords:",len(keywords),file=sys.stderr)
corpuswords = {}
print("Reading corpus words and their idf ...",file=sys.stderr)
with open(corpusWordsFile) as inp:
    for line in inp:
        line = line.strip()
        (word,idf) = line.split("\t")
        corpuswords[word] = float(idf)
print("INFO: got corpus words:",len(corpuswords),file=sys.stderr)

print("Calculating similarities, this will take a while ...",file=sys.stderr)

## Implementation note:
## * corpus words are filtered/ignored if the corpus word is identical or a word in a key phrase
## * embedding similarity is calculated, corpus word is ignored if sim < MIN_EMB_SIM
## * the following scores are calculated for the pair:
##   * raw embedding sim (sim)
##   * embedding sim times corpus idf (simidf)
##   * embedding sim times corpus idf times inverse string sim (simidfstr)
## * three ordered lists for each of the scores are maintained
## * the highest k corpus words for each of the three scores are output, together
##   with their rank numbers

# k is the number of corpus words that corresponds to KEEP_FRACTION
k = int(len(corpuswords)*KEEP_FRACTION)
totalpairs = 0
written = 0
with tqdm(total=(len(corpuswords)*len(keywords))) as pbar:
    for keyword in keywords:
        h_sim = []
        h_simidf = []
        h_simidfstr = []
        for corpusword,idf in corpuswords.items():
            pbar.update(1)
            totalpairs = totalpairs + 1
            # filter words we are not interested in
            # 1) the corpus word is equal to the keyword according to the embeddings
            (embsim,usedcorpuswords,usedkeywords) = eu.sim4texts(corpusword,keyword)
            # print("DEBUG embsim for",corpusword,"/",keyword,"=",embsim,file=sys.stderr)
            if np.isclose(embsim, 1.0, rtol=1e-07, atol=1e-09, equal_nan=False):
                if debug: print("DEBUG: skipping equal pair",corpusword,keyword,file=sys.stderr)
                continue
            # 2) the minimum embedding similarity threshold is not reached
            if embsim < MIN_EMB_SIM:
                #if debug: print("DEBUG: skipping low sim: ",corpusword,"/",keyword,"=",embsim,file=sys.stderr)
                continue
            # 3) one of the (used) corpuswords is in the keyword words
            isContained = False
            for w in usedcorpuswords:
               if w in usedkeywords:
                   isContained = True
                   break
            if isContained:
                if debug: print("DEBUG: skipping, isContained",corpusword,"/",keyword,"usedcorp=","|".join(usedcorpuswords),"usedkey=","|".join(usedkeywords),file=sys.stderr)
                continue
            # 4) if the original keyphrase was significantly shortened ignore this
            tokens = eu.tokens4text(keyword)
            if len(tokens) > 1 and usedkeywords == 1:
                if debug: print("DEBUG: skipping shrunk keyword: ",tokens,usedkeywords,file=sys.stderr)
                continue
            # ok, we want to keep this pair, store the corpusword in the heaps, for their scores
            # print("DEBUG: pushing ",(embsim,corpusword),file=sys.stderr)
            heapq.heappush(h_sim,(embsim,corpusword,usedcorpuswords,usedkeywords))
            simidf = embsim * idf
            heapq.heappush(h_simidf,(simidf,corpusword,usedcorpuswords,usedkeywords))
            stringsim = SequenceMatcher(None,corpusword,keyword).ratio()
            simidfstr = embsim * idf * (1.0/(stringsim+1))
            heapq.heappush(h_simidfstr,(simidfstr,corpusword,usedcorpuswords,usedkeywords))
        # after going through all the corpuswords, output the best k
        if len(h_sim) == 0:
            if debug: print("DEBUG: nothing for",keyword,file=sys.stderr)
            pass
        else:
            # print("DEBUG: heap is ",h_sim,file=sys.stderr)
            l1 = heapq.nlargest(k,h_sim)
            rank = 0
            for s,w,cws,kws in l1:
               print("simonly",w,keyword,rank,s,cws,kws,sep="\t")
               written = written + 1
               rank = rank + 1
            l1 = heapq.nlargest(k,h_simidf)
            rank = 0
            for s,w,cws,kws in l1:
               print("simidfonly",w,keyword,rank,s,cws,kws,sep="\t")
               written = written + 1
               rank = rank + 1
            l1 = heapq.nlargest(k,h_simidfstr)
            rank = 0
            for s,w,cws,kws in l1:
               print("simidfstr",w,keyword,rank,s,cws,kws,sep="\t")
               written = written + 1
               rank = rank + 1

print("Finished, total comparisons:",totalpairs,"written:",written,file=sys.stderr)
