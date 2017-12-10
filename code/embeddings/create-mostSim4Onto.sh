#!/bin/bash

## create the mostSim4Onto files for all 4 embeddings

python3 mostSim4Onto.py ../../data/knowmak-ontology.lst corpuswords-facebook.tsv ../../embeddings/wiki.en.gensim 0 > mostSim4Onto-facebook.tsv

python3 mostSim4Onto.py ../../data/knowmak-ontology.lst corpuswords-glove.6B.tsv ../../embeddings/glove.6B.300d.gensim 0 > mostSim4Onto-glove.6B.tsv

python3 mostSim4Onto.py ../../data/knowmak-ontology.lst corpuswords-glove.840B.tsv ../../embeddings/glove.840B.300d.gensim 1 > mostSim4Onto-glove.840B.tsv

python3 mostSim4Onto.py ../../data/knowmak-ontology.lst corpuswords-googlenews.tsv ../../embeddings/GoogleNews-vectors-negative300.gensim 1 > mostSim4Onto-googlenews.tsv
