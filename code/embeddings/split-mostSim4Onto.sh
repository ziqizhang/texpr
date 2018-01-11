#!/bin/bash

infile="$1"
maxrank="$2"
if [[ "x$maxrank" == "x" ]]
then 
  echo need two parameters, the name of the input file and max rank
  exit 1
fi

inpref=`basename $infile .tsv`
#grep "simonly" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2- | sort -t$'\t' --key=9nr,9  > ${inpref}-sim.tsv
#grep "simidfonly" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2- | sort -t$'\t' --key=9nr,9  > ${inpref}-simidf.tsv
#grep "simidfstr" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2- | sort -t$'\t' --key=9nr,9  > ${inpref}-simidfstr.tsv

## to make debugging easier, do not sort by final score
grep "simonly" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2-    > ${inpref}-sim-${maxrank}.tsv
ncw=`cat ${inpref}-sim-${maxrank}.tsv | cut -f 1 | sort -u | wc -l`
nkw=`cat ${inpref}-sim-${maxrank}.tsv | cut -f 2 | sort -u | wc -l`
echo SIM corpuswords $ncw keywords $nkw
grep "simidfonly" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2- > ${inpref}-simidf-${maxrank}.tsv
ncw=`cat ${inpref}-simidf-${maxrank}.tsv | cut -f 1 | sort -u | wc -l`
nkw=`cat ${inpref}-simidf-${maxrank}.tsv | cut -f 2 | sort -u | wc -l`
echo SIMIDF corpuswords $ncw keywords $nkw
grep "simidfstr" $infile | python3 postprocessMostSim4Onto.py $maxrank | cut -f 2-  > ${inpref}-simidfstr-${maxrank}.tsv
ncw=`cat ${inpref}-simidfstr-${maxrank}.tsv | cut -f 1 | sort -u | wc -l`
nkw=`cat ${inpref}-simidfstr-${maxrank}.tsv | cut -f 2 | sort -u | wc -l`
echo SIMIDFSTR corpuswords $ncw keywords $nkw


