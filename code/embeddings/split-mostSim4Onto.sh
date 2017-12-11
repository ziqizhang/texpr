#!/bin/bash

infile="$1"
if [[ "x$infile" == "x" ]]
then 
  echo need one parameter, the name of the input file
  exit 1
fi

inpref=`basename $infile .tsv`
grep "simonly" $infile | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-sim.tsv
grep "simidfonly" $infile | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-simidf.tsv
grep "simidfstr" $infile | cut -f 2- | sort -t$'\t' --key=4nr,4  > ${inpref}-simidfstr.tsv

