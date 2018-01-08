

#this method parses the REL task dataset from http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/downloads.shtml
#to extract named entities
import os
import re

regex = re.compile('[^a-zA-Z]')
in_folder="/home/zqz/Work/data/BioNLP-ST 2011/merged"
out_file="/home/zqz/Work/data/texpr/dict/bio_2011REL.txt"
candidates=set()
for in_file in os.listdir(in_folder):
    if not in_file.endswith(".a1") and not in_file.endswith(".rel"):
        continue
    with open(in_folder+"/"+in_file, encoding="utf8") as f:
        lines = f.readlines()
        for l in lines:
            parts=l.split("\t")
            if len(parts)<3:
                continue
            name=parts[len(parts)-1].strip()
            alpha=regex.sub('', name)
            if len(alpha)<3:
                continue
            candidates.add(name)
sorted = list(candidates)
sorted.sort()
with open(out_file, 'w') as the_file:
    for e in sorted:
        the_file.write(e+'\n')
