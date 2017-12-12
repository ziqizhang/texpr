#!/bin/bash

infile="$1"
maxrank="$2"
if [[ "x$maxrank" == "x" ]]
then 
  echo need two parameters, the name of the input file and max rank
  exit 1
fi

inpref=`basename $infile .tsv`
grep "simonly" $infile | perl -F'\t' -ane "if (\$F[3] < $maxrank) {print}" | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-sim.tsv
grep "simidfonly" $infile | perl -F'\t' -ane "if (\$F[3] < $maxrank) {print}" | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-simidf.tsv
grep "simidfstr" $infile | perl -F'\t' -ane "if (\$F[3] < $maxrank) {print}" | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-simidfstr.tsv

