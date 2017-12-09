import json
import re
from threading import Thread

import datetime
import numpy
from texpr.sim import seed_selector as ss
from embeddings.embeddingsutils import EmbeddingsUtils


# Define a function for the thread
def calc_sim(gs_list: list, candidate_list: list, thread_id, embu: EmbeddingsUtils,
             output: dict, min_score=0.0):
    count_gt = 0
    print("\tTHREAD {}, gs has {}, candidate has {}".format(thread_id, str(len(gs_list))
                                                            , str(len(candidate_list))))
    for gt in gs_list:
        count_gt += 1
        count_ct = 0
        for ct in candidate_list:
            count_ct += 1
            res = embu.sim4texts(gt, ct)
            score = res[0]
            if score == 0.0 or score<min_score:
                continue
            if res[1] in output.keys():
                output[res[1]].append({res[2]: str(score)})
            else:
                sim_scores = []
                sim_scores.append({res[2]: str(score)})
                output[res[1]] = sim_scores
            if count_ct % 5000 == 0:
                print("\t\t{} thread {}, gs {}/{}, ct {}/{}".format(
                    datetime.datetime.now(), thread_id,
                    str(count_gt), str(len(gs_list)),
                    str(count_ct), str(len(cand_list))))

    return output


def read_list(inFile: str):
    regex = re.compile('[^a-zA-Z]')
    output = set()
    if inFile.endswith(".txt"):
        with open(inFile, encoding="utf8") as f:
            for line in f:
                output.update(regex.sub(' ', line.strip()).split(" "))
    elif inFile.endswith(".json"):
        with open(inFile, encoding="utf8") as json_data:
            d = json.load(json_data)
            for item in d:
                output.update(regex.sub(' ', item["string"]).split(" "))
    output.remove("")
    return output


# EMBEDDING_FILE="/home/zqz/Work/data/GoogleNews-vectors-negative300.bin.gz"
EMBEDDING_FILE = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
INPUT_GS_LIST = "/home/zqz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2.txt"
INPUT_CANDIDATE_LIST = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
SIM_OUT_FILE = "/home/zqz/Work/data/texpr/sim_genia.json"
THREADS = 4
SELECT_GS = 1000

#######################
print("selecting randomly {} from GS list..".format(str(SELECT_GS)))
gs_list = ss.select_random_from_list(INPUT_GS_LIST, SELECT_GS)
print("loading candidate words {} from candidate term list..".format(str(SELECT_GS)))
cand_list = read_list(INPUT_CANDIDATE_LIST)
threads = THREADS
gs_list_chunks = numpy.array_split(gs_list, threads)

eu = EmbeddingsUtils()
eu.setIsCaseSensitive(False)
eu.setFallBackToLower(True)
eu.setFilterStopwords(True)
# eu.setDebug(debug)
# eu.setVerbose(verbose)
eu.loadEmbeddings(EMBEDDING_FILE)
# exit(0)


# Create two threads as follows

id = 0
threads = []
outputs = []
for chunk in gs_list_chunks:
    o = dict()
    outputs.append(o)
    t = Thread(target=calc_sim, args=(chunk, cand_list, id, eu, o))
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

with open(SIM_OUT_FILE, 'w', encoding='utf8') as ofile:
    json.dump(final_dict, ofile)
