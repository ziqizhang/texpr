#!/bin/bash

infile="$1"
if [[ "x$infile" == "x" ]]
then 
  echo need one parameter, the name of the input file
  exit 1
fi

inpref=`basename $infile .tsv`
grep -v "sim\*idf" $infile | cut -f 2- | sort -k 3nr > ${inpref}-sim.tsv
grep  "sim\*idf" $infile | cut -f 2- | sort -k 3nr > ${inpref}-simidf.tsv

