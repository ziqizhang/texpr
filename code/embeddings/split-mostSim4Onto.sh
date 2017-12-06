#!/bin/bash

grep -v "sim\*idf" mostSim4Onto.tsv | cut -f 2- | sort -k 3nr > mostSim4Onto-sim.tsv
grep  "sim\*idf" mostSim4Onto.tsv | cut -f 2- | sort -k 3nr > mostSim4Onto-simidf.tsv

