from __future__ import print_function
import sys
import re
from difflib import SequenceMatcher

## simple script to filter the following rows from the mostSim4Onto tsv files:
## * columns 1 and 2 are equal
## * column 1 is a substring of column 2

for line in sys.stdin:
  line = line.rstrip()
  fields = line.split("\t")
  sim = float(fields[2])
  # there are also some odd cases where punctuation sticks to the end of the
  # fields[0], lets remove that first
  fields[0] = re.sub("[.,;:'\"?!]+","",fields[0]) 
  if len(fields) != 3:
    print("Odd input line, got fields: ",len(fields),", line=",line,file=sys.stderr)
    sys.exit(1)
  if fields[0] == fields[1] or fields[0] in fields[1]:
    ## skip this one
    pass
  else:
    stringsim = SequenceMatcher(None,fields[0],fields[1]).ratio()
    print(line,stringsim,(1.0/stringsim)*sim,sep="\t")


