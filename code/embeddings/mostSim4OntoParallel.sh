#!/bin/bash

# usage: mostSim4OntoParallel.sh keywords corpuswords embs caseflag outputfile
kwfile="$1"
cwfile="$2"
embfile="$3"
caseflag="$4"
stopwords="$5"
outfile="$6"

if [ "x$outfile" == "x" ]
then
  echo "usage: mostSim4OntoParallel.sh keywords corpuswords embs caseflag stopwordsfile outputfile"
  exit 1
fi

ncores=`nproc --all`
echo Running $ncores copies in parallel
lines=`wc -l < $kwfile`
echo keywords file has $lines lines
perfile=$((lines / ncores))
echo expected keywords per thread about $perfile
rm tmp-keywords*
rm tmp-output-*
split -d -e -n $ncores "$kwfile" tmp-keywords
wc -l tmp-keywords*

# to be able to track the progress of all threads we start each thread in its own
# terminal
for kw in tmp-keywords*
do
  echo 'starting xterm -e python3 mostSim4Onto.py $kw termraider-words.tsv ../../embeddings/glove.840B.300d.gensim $caseflag $stopwords > tmp-output-$kw'
  xterm -e "echo running for $kw; python3 mostSim4Onto.py $kw termraider-words.tsv ../../embeddings/glove.840B.300d.gensim $caseflag $stopwords > tmp-output-$kw" &
done
echo waiting for all to complete
wait
echo finalising the outputs
cat tmp-output-* > $outfile
# concatanate the output
