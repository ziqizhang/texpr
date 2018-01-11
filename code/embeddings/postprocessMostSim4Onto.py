from __future__ import print_function
import sys
import re
from difflib import SequenceMatcher

## simple script to filter and postprocess the mostSim4Onto result file
## * expect one numeric argument: maximum rank number, everything larger gets filtered
## * add one column at the end: product of columns 5 (score) and 9 (keyword average idf)
## (0-based these are cols 4 and 8)

if len(sys.argv) != 2:
  print("ERROR: need one parameter, maximum rank number to include",file=sys.stderr)
  sys.exit(1)

maxrank = int(sys.argv[1])
filtered = 0
n = 0
nout = 0
maxrankfound = 0
for line in sys.stdin:
  line = line.rstrip()
  fields = line.split("\t")
  if len(fields) != 9:
    print("Odd input line, got fields: ",len(fields),", line=",line,file=sys.stderr)
    sys.exit(1)
  simid = fields[0]
  n += 1
  rank = int(fields[3])
  score = float(fields[4])
  kwidf = float(fields[8])
  if rank > maxrankfound:
    maxrankfound = rank
  if rank > maxrank:
    filtered += 1
    continue
  newscore = score * kwidf
  print(line,newscore,sep="\t")
  nout += 1

print("FILTERED lines for",simid,"=",filtered,"of total",n,"written",nout,"maxrankfound=",maxrankfound,file=sys.stderr)


