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

from __future__ import print_function
import sys
import os
from embeddingsutils import EmbeddingsUtils
from tqdm import tqdm

if len(sys.argv) != 4:
    print("ERROR: need the following arguments: keywords lst file, corpus words tsv file, embeddings path",file=sys.stderr)
    sys.exit(1)

keywordsFile = sys.argv[1]
corpusWordsFile = sys.argv[2]
embFile = sys.argv[3]

eu = EmbeddingsUtils()
eu.setIsCaseSensitive(False)
eu.setFallBackToLower(False)
eu.setFilterStopwords(True)
eu.setDebug(False)
eu.setVerbose(False)
eu.loadEmbeddings(embFile)

keywords = set()
print("Reading keywords ...",file=sys.stderr)
with open(keywordsFile) as inp:
    for line in inp:
        (keyword,cname,other) = line.split("\t")
        ## cname = cname[4:]
        keywords.add(keyword)

corpuswords = set()
print("Reading keywords ...",file=sys.stderr)
with open(corpusWordsFile) as inp:
    for line in inp:
        (word,idf) = line.split("\t")
        corpuswords.add(word)

print("Calculating similarities...",file=sys.stderr)

with tqdm(total=len(corpuswords)) as pbar:
    for word in corpuswords:
        bestkeyword = None
        bestsim = -1.0
        for keyword in keywords:
            (sim,used1,used2) = eu.sim4texts(keyword,word)
            if sim > bestsim:
                bestkeyword = keyword
                bestsim = sim
        print(word,bestsim,sep="\t")
        pbar.update(1)
