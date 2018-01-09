#!/bin/bash

# convert the ontology gazetteer file to the classinfo file needed by the finalRanking.py script
# This expects gazetter .lst format with the following tab-separated fields in the following order
# 1 - keyword
# 2 - uri=<uri>
# 3 - topicId=<topicid>
# 4 - flags=<flags>
# 5 - provenance=<provenance>
# 6 - property=<property>

# This will create a tsv file with the following fields:
# 1 - keyword
# 2 - uri
# 3 - section: either SGC or KET, dependent on the provenance field

cat | cut -f 1,2,5 | sed -e 's/uri=//' -e 's/provenance=eupro-classes.xlsx/SGC/' -e 's/provenance=Nature/KET/' -e 's/provenance=KET-top-level-taxonomy.xlsx/KET/' -e 's/provenance=SGC-IPC-mapping.xlsx + ipc.xlsx/SGC/' -e 's/provenance=SGC_taxonomy_tree.xlsx/SGC/' 
