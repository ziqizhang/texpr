import json
import re
from threading import Thread

import datetime
import numpy
from texpr.sim import seed_selector as ss
from embeddings.embeddingsutils import EmbeddingsUtils

MAX_DICT_SIZE_BEFORE_DUMPING = 500


# Define a function for the thread
def calc_sim(gs_list: list, candidate_list: list, thread_id, embu: EmbeddingsUtils,
             output: dict, out_folder, data_label, min_score=0.0, topnperc=1.0):
    count_gt = 0
    count_batch = 0
    print("\tTHREAD {}, gs has {}, candidate has {}".format(thread_id, str(len(gs_list))
                                                            , str(len(candidate_list))))
    topn = int(len(cand_list) * topnperc)

    for gt in gs_list:
        count_gt += 1
        count_ct = 0
        for ct in candidate_list:
            count_ct += 1
            if count_ct % 2000 == 0:
                print("\t\t{} thread {}, gs {}/{}, ct {}/{}".format(
                    datetime.datetime.now(), thread_id,
                    str(count_gt), str(len(gs_list)),
                    str(count_ct), str(len(cand_list))))
            res = embu.sim4texts(gt, ct)
            score = res[0]
            if score == 0.0 or score < min_score:
                continue
            if res[1][0] in output.keys():
                output[res[1][0]].append((res[2][0], str(score)))
            else:
                sim_scores = []
                sim_scores.append((res[2][0], str(score)))
                output[res[1][0]] = sim_scores

        # if count_gt==1:
        #     break
        if count_gt % MAX_DICT_SIZE_BEFORE_DUMPING==0:
            print("\t\t\t{} thread {}, saving batch {} with size={}".format(
                datetime.datetime.now(), thread_id,
                str(count_batch),str(len(output))))
            count_batch+=1
            resized=resize(output,topn)
            with open(out_folder+"/"+data_label+str(thread_id)+"-"+str(count_batch)+".json", 'w', encoding='utf8') as ofile:
                json.dump(resized, ofile)
            output.clear()

    return output


def resize(dict:dict, topn):
    new_dict={}
    for key,val in dict.items():
        val.sort(key=lambda x: x[1], reverse=True)
        shortened=val[:topn]
        new_dict[key]=shortened
    return new_dict


def read_list(inFile: str):
    regex = re.compile('[^a-zA-Z]')
    output = set()
    if inFile.endswith(".txt"):
        with open(inFile, encoding="utf8") as f:
            for line in f:
                entry=line.strip().split("\t")[0]
                if len(entry)<3:
                    continue
                output.update(regex.sub(' ', entry).split(" "))
    elif inFile.endswith(".json"):
        with open(inFile, encoding="utf8") as json_data:
            d = json.load(json_data)
            for item in d:
                output.update(regex.sub(' ', item["string"]).split(" "))
    #output.remove("")
    return output


EMBEDDING_FILE = "/home/zqz/Work/data/glove.840B.300d.bin.gensim"
# EMBEDDING_FILE = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
INPUT_GS_LIST = "/home/zqz/Work/data/texpr/dict/bio_2011REL.txt"
INPUT_CANDIDATE_LIST = "/home/zqz/Work/data/texpr/corpus_words/words_genia.txt"
SIM_OUT_FOLDER = "/home/zqz/Work/data/texpr"
DATA_LABEL = "genia"
THREADS = 4
SELECT_GS = 50000

# EMBEDDING_FILE = "/home/zqz/Work/data/glove.840B.300d.bin.gensim"
# # EMBEDDING_FILE = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# INPUT_GS_LIST = "/home/zqz/Work/data/texpr/dict/acl_anthology.txt"
# INPUT_CANDIDATE_LIST = "/home/zqz/Work/data/texpr/corpus_words/words_acl.txt"
# SIM_OUT_FOLDER = "/home/zqz/Work/data/texpr"
# DATA_LABEL = "acl"
# THREADS = 4
# SELECT_GS = 50000

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
    t = Thread(target=calc_sim, args=(chunk, cand_list, id, eu, o,
                                      SIM_OUT_FOLDER, DATA_LABEL, 0.0, 0.2))
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

with open(SIM_OUT_FOLDER+"/"+DATA_LABEL+".json", 'w', encoding='utf8') as ofile:
    json.dump(final_dict, ofile)
