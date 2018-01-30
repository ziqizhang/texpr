
# Simple script to calculate the idf from raw document count in a termraider file and
# output a tsv file. Will also do some initial cleaning/filtering of the terms.
#
# Input format: CSV, first row is header, second row contains total number of documents (in last/5th column)
# subsequent rows have columns
# * term
# * language
# * type (multi/single word)
# * document frequency
#
# Output will be tsv file without a header with columns
# * term
# * idf (calculated as log(N/n) N=total, n=with term (n will never be 0 so no smoothing needed)

## Author: Johann Petrak <johann.petrak@gmail.com>

from __future__ import print_function
import sys
import argparse
import math
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description="Convert termraider documentfreq csv to corpus words/terms+idf tsv")
parser.add_argument("-v", action='store_true', help="Show more messages about what the program is doing.")
parser.add_argument("-d", action='store_true', help="Show debug information.")
parser.add_argument("-s", nargs=1, type=str, help="File to use for removing stopwords")

args = parser.parse_args()

debug = args.d
verbose = args.v or debug
stops = set(stopwords.words("english"))
if args.s:
    stopFile = args.s[0]
    with open(stopFile) as infile:
        for line in infile:
            line = line.rstrip()
            fields = line.split("\t")
            stopword = fields[0]
            stops.add(stopword)


nInput = 0
nOutput = 0
totalDocs = 0
for line in sys.stdin:
    nInput = nInput + 1
    line = line.rstrip("\n")
    fields = line.split(",")
    # handle the two first lines differently
    if nInput == 1:
        pass # simply ignore
    elif nInput == 2:
        totalDocs = float(fields[-1])
    else:
        term = fields[0].strip()
        if term in stops:
            continue
        freq = float(fields[3])
        # TODO: cleaning and filtering would go here
        print(term, math.log(totalDocs/freq), sep="\t")
        nOutput = nOutput + 1
print("Input lines: ", nInput, file=sys.stderr)
print("Output lines: ", nOutput, file=sys.stderr)
