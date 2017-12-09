#!/bin/bash


## calculate the words for the corpus for each of the 4 embeddings files we consider

## Glove 6B: lower case only
## words: 29101
## OOV: 21946
python3 corpus2words.py ../../embeddings/glove.6B.300d.gensim ../../corpora/policy-docs-plain-text/ oov-glove.6B.json 0 > corpuswords-glove.6B.tsv


## Glove 840B: mixed case
## Using lower case:
## words: 31285
## OOV: 19762
## Using mixed case with fallback:
## words: 44419
## OOV: 16139
python3 corpus2words.py ../../embeddings/glove.840B.300d.gensim ../../corpora/policy-docs-plain-text/ oov-glove.840B.json 1 > corpuswords-glove.840B.tsv

## Googlenews: mixed case
## words: 22360
## OOV: 28687
## Using mixed case with fallback:
## words: 37522
## OOV: 22255
python3 corpus2words.py ../../embeddings/GoogleNews-vectors-negative300.gensim ../../corpora/policy-docs-plain-text/ oov-googlenews.json 1 > corpuswords-googlenews.tsv

## Facebook: lowercase
## words: 31986
## OOV: 19061
python3 corpus2words.py ../../embeddings/wiki.en.gensim ../../corpora/policy-docs-plain-text/ oov-facebook.json 0 > corpuswords-facebook.tsv

