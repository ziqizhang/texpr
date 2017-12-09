
## Simple script to calculate the embedding-based similarity between
## two texts coming from the columns of a TSV file.
## * -a gives the column number where to get the first text from (0-based, default 0)
## * -b gives the column number where to get the second text from
## * -e gives the path to an embeddings file or file prefix
## * reads from standard input, writes 3-column (or 4-column if id column is also given) tsv file to standard output
## * acts as a pipe and reads from stadin and writes output to stdout, errors etc to stderr
## * similarity is calculated as the cosine similarity of averaged embeddings for known words
##   if all words of a text are not known, the similarity is 0.0
## NOTE: this probably only works with python 3.5 or later

## Author: Johann Petrak <johann.petrak@gmail.com>

from __future__ import print_function
import sys
import argparse
from embeddingsutils import EmbeddingsUtils

parser = argparse.ArgumentParser(description="Find similarity between texts")
parser.add_argument("-v", action='store_true', help="Show more messages about what the program is doing.")
parser.add_argument("-d", action='store_true', help="Show debug information.")
parser.add_argument("-c", action='store_true', help="Make the processing case-sensitive (default: insensitive)")
parser.add_argument("-B", action='store_true', help="Backup to lower-case if word not lower case and OOV (only if -c)")
parser.add_argument("-e", nargs=1, type=str, help="The embedding files to read.")
parser.add_argument("-a", nargs=1, type=int, help="first input column number, 0-based (default 0)")
parser.add_argument("-b", nargs=1, type=int, help="second input column number, 0-based (default 1)")
parser.add_argument("-k", nargs=1, type=int, help="column number, 0-based of additional (key/id) column to copy over, default is none")
parser.add_argument("--stop", action='store_true', help="Include (do not remove) English stop words")
parser.add_argument("--mo", action='store_true', help="Use multiplicative objective instead of cosine")
parser.add_argument("--oov", nargs=1, type=str, help="Store OOV information in the file given (default: print to stderr)")

args = parser.parse_args()

verbose = args.v
debug = args.d

if args.e:
    embFile = args.e[0]
else:
    print("ERROR: need option -e to specify the embeddings file",file=sys.stderr)
    sys.exit(1)

input1 = 0
if args.a:
    input1 = args.a[0]
input2 = 1
if args.b:
    input2 = args.b[0]

copyOver = None
if args.k:
    copyOver = args.k[0]

outOOV = ""
if args.oov:
  outOOV = args.oov[0]
includestops = args.stop

useMultiObj = args.mo
if useMultiObj:
    print("Using multiplicative objective instead of cosine",file=sys.stderr)

caseSensitive = args.c
backupToLower = args.B

missingWords = {}
nEmpty = 0

eu = EmbeddingsUtils()
eu.setIsCaseSensitive(caseSensitive)
eu.setFallBackToLower(backupToLower)
eu.setFilterStopwords(not includestops)
eu.setDebug(debug)
eu.setVerbose(verbose)
eu.loadEmbeddings(embFile)

nInput = 0
nOutput = 0
for line in sys.stdin:
    nInput = nInput + 1
    if verbose and nInput % 1000 == 0:
        print("Lines processed: ", nInput, file=sys.stderr)
    line = line.rstrip("\n")
    fields = line.split("\t")
    text1 = fields[input1]
    text2 = fields[input2]
    ## now calculate the similarity
    (sim,t1,t2) = eu.sim4texts(text1,text2)
    if copyOver:
        print(fields[copyOver],end="\t")
    print(text1, t1, text2, t2, sim, sep="\t")
    nOutput = nOutput + 1
print("Input lines: ", nInput, file=sys.stderr)
print("Output lines: ", nOutput, file=sys.stderr)
print("Number of inputs mapped to empty strings: ", nEmpty, file=sys.stderr)
print("Number of OOV words: ",len(missingWords), file=sys.stderr)
if outOOV:
    # save the OOV information to a file
    with open(outOOV,'w') as outoov:
      for missingword in eu.missingWords.keys():
        print(missingword,eu.missingWords[missingword],file=outoov,sep="\t")
else:
    print("OOV words: ", ", ".join(eu.missingWords.keys()), file=sys.stderr)
