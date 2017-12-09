## convert word embeddings file to gensim's fast binary format
## NOTE: maybe we should consider exporting using fast serialization as described here:
##   http://arrow.apache.org/blog/2017/10/15/fast-python-serialization-with-ray-and-arrow/
## See https://github.com/apache/arrow 

## Author: Johann Petrak <johann.petrak@gmail.com>


import sys
import gensim
from gensim.models.word2vec import Vocab
import numpy as np
import scipy as sp
import argparse


parser = argparse.ArgumentParser(description="Convert embeddings to gensim binary format")
parser.add_argument("-v", action='store_true', help="Show more messages about what the program is doing.")
parser.add_argument("embFile", nargs=1, type=str, help="The embedding files to read.")
parser.add_argument("saveFile", nargs=1, type=str, help="The file to save to, must have extension .gensim")

args = parser.parse_args()

verbose = args.v

embFile = args.embFile[0]
saveFile = args.saveFile[0]

if not saveFile.endswith(".gensim"):
    print("Output file must have extension .gensim", file=sys.stderr)
    sys.exit(1)

inBinary = ".bin" in embFile 
if verbose: print("Loading embeddings file ", embFile," using binary format: ",inBinary,file=sys.stderr)
model = gensim.models.KeyedVectors.load_word2vec_format(embFile, binary=inBinary, unicode_errors='ignore', encoding='utf8')
print("Embeddings loaded, words: ",len(model.index2word),file=sys.stderr)
print("Calculating norm and saving...",file=sys.stderr)
model.init_sims(replace=True)
model.save(saveFile)  
print("Saved!",file=sys.stderr)

