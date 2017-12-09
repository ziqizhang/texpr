#!/bin/bash

if [[ "x${EMBEDDINGS_DIR}" == "x" ]]
then
  echo 'Environment variable EMBEDDINGS_DIR should be set to where the embeddings file is stored'
  exit 1
fi

# use python3!
python mostSim4Onto.py ../../data/knowmak-ontology.lst corpuswords.tsv ${EMBEDDINGS_DIR}/glove.840B.300d.gensim > mostSim4Onto.tsv
