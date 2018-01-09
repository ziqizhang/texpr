#!/bin/bash

# read in the knowmak ontology gazetteer file and create a keyword file 
# for use with mostSim4Onto
# Works as a pipe

cat | cut -f 1 | sed -e 's/-/ /g' -e 's/—/ /g' | sort -u 
