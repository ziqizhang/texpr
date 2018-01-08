#computes similarity between each candidate term and the reference dictionary, find average score
#the average is over the dict entries with which sim is non-zero
import datetime
import json
import re
import sys
from threading import Thread

import numpy

from embeddings import EmbeddingsUtils


def calc_sim(dict_list: list, candidate_list: list, thread_id, embu: EmbeddingsUtils,
             output:dict):
    count_ct = 0
    print("\tTHREAD {}, dict has {}, candidate has {}".format(thread_id, str(len(dict_list)),
                                                              len(candidate_list)))
    for ct in candidate_list:
        count_ct += 1
        count_dict = 0
        count_dict_nonzero_sim=0
        sum_sim=0
        for dt in dict_list:
            count_dict += 1
            if count_dict % 1000 == 0:
                print("\t\t{} thread {}, ct {}/{}, dict {}/{}".format(
                    datetime.datetime.now(), thread_id,str(count_ct), str(len(candidate_list)),
                    str(count_dict), str(len(dict_list))
                    ))
            res = embu.sim4texts(ct, dt)
            score = res[0]
            if score == 0.0:
                continue
            count_dict_nonzero_sim+=1
            sum_sim+=score

        if count_dict_nonzero_sim>0:
            final_score=sum_sim/count_dict_nonzero_sim
        else:
            final_score=0.0
        output[ct]=final_score

    return output



#####################################
#/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json
#/home/zqz/Work/data/texpr/dict/acl_anthology.txt
#/home/zqz/Work/data/glove.840B.300d.bin.gensim
#/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/texpr_base.json

REGEX = re.compile('[^a-zA-Z\-]')
candidate_list=[]
data = json.load(open(sys.argv[1], encoding="utf8"))
for item in data:
    candidate_list.append(item["string"])

dict_list=[]
with open(sys.argv[2], encoding='utf8') as f:
    for line in f:
        processed = REGEX.sub(' ', line.strip())
        processed = re.sub(' +', ' ', processed)
        if len(processed) > 3:
            dict_list.append(processed)

threads = 4
cand_list_chunks = numpy.array_split(candidate_list, threads)

eu = EmbeddingsUtils()
eu.setIsCaseSensitive(False)
eu.setFallBackToLower(True)
eu.setFilterStopwords(True)
# eu.setDebug(debug)
# eu.setVerbose(verbose)
eu.loadEmbeddings(sys.argv[3])
# exit(0)

# Create two threads as follows

id = 0
threads = []
outputs = []
for chunk in cand_list_chunks:
    o = dict()
    outputs.append(o)
    t = Thread(target=calc_sim, args=(dict_list, chunk, id, eu,o))
    threads.append(t)
    id += 1

for t in threads:
    t.start()
for t in threads:
    t.join()

print("all done")

final_dict = outputs[0]
for i in range(1, 4):
    final_dict.update(outputs[i])

with open(sys.argv[4], 'w', encoding='utf8') as ofile:
    json.dump(final_dict, ofile)
