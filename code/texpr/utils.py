import csv
import json
import os
import tarfile
import re
import numpy as np

import pickle as pk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("init")#this forces lemmatizer to load, in order to avoid non-thread safe usage of lemmatizer

WORD_EMBEDDING_VOCAB_MIN_LENGTH=3

class GeniaCorpusLoader(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        tar = tarfile.open(self.dirname, "r:gz")
        count = 0
        for member in tar.getmembers():
            count = count + 1
            f = tar.extractfile(member)
            # print("progress:", count, member
            #     )
            if f is None:
                continue
            try:
                content = f.read().decode('utf8').strip().replace('\n', '')
                # content = the_file.read().replace('\n', '')
                sents = sent_tokenize(content)
                for sent in sents:
                    yield normalize_string(sent)
            except ValueError:
                print("cannot process: {}".format(content))


def randomize(semrerank):
    values=np.random.randn(len(semrerank))
    print("randomize {} unigrams...".format(len(semrerank)))
    index = 0
    for key in semrerank.keys():
        semrerank[key] = values[index]
        index += 1
    semrerank = normalize(semrerank)
    return semrerank


def normalize_string(original):
    toks = word_tokenize(original)
    norm_list = list()
    for tok in toks:
        # if tok=="b_alpha" or 'dihydroxycholecalciferol' in tok:
        #     print("caught")
        lem = lemmatizer.lemmatize(tok).strip().lower()
        lem = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', lem).strip()
        for part in lem.split():
            if keep_token(lem):
                norm_list.append(part)
    # sent= " ".join(norm_list)
    return norm_list


def keep_token(token):
    token = re.sub(r'[^a-zA-Z]', '', token).strip()
    return len(token)>=WORD_EMBEDDING_VOCAB_MIN_LENGTH


def read_and_normalize_terms(filePath):
    list=[]
    with open(filePath, encoding="utf-8") as file:
        terms = file.readlines()
        for t in terms:
            t = lemmatizer.lemmatize(t.strip()).strip().lower()
            t = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', t).strip()
            list.append(t)
    return list


def find_ngrams(input_list, n):
    # out = list()
    # for num in range(1, n):
    #     if (num > len(input_list)):
    #         break
    #     candidates = list(ngrams(input_list, num))
    #     for ng in candidates:
    #         out.append("_".join(ng))
    # return out
    return input_list


def normalize(d):
    max = 0.0
    for key, value in d.items():
        if value > max:
            max = value
    return {key: value / max for key, value in d.items()}


def load_saved_model(file):
    if os.path.isfile(file):
        file = open(file, 'rb')
        return pk.load(file)
    else:
        return None


def read_lines(file):
    array = []  # declaring a list with name '**array**'
    with open(file, 'r') as reader:
        for line in reader:
            array.append(line.strip())
    return array


def genia_corpus_to_unigrams(input_gz_file, out_folder):
    tar = tarfile.open(input_gz_file, "r:gz")
    count = 0
    for member in tar.getmembers():
        count = count + 1
        f = tar.extractfile(member)
        # print("progress:", count, member
        #     )
        if f is None:
            continue

        unigrams=set()
        try:
            content = f.read().decode('utf8').strip().replace('\n', '')
            # content = the_file.read().replace('\n', '')
            sents = sent_tokenize(content)
            for sent in sents:
                unigrams.update(normalize_string(sent))
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")

        file=member.name
        new_file_path=out_folder+"/"+str(file)
        path = Path(new_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(new_file_path, 'w')
        for ug in unigrams:
            out_file.write("%s\n" % ug)


def jate_terms_iterator(jate_json_outfile):
    #logger.info("Loading extracted terms by JATE...")
    json_data = open(jate_json_outfile).read()
    data = json.loads(json_data)
    count = 0

    if type(data) is list:
        for term in data:
            count = count + 1
            yield term['string'], term['score']
    else:
        for k,v in data.items():
            yield k, v
        # if (count % 2000 == 0):
        #     logger.info("\t loaded {}".format(count))
        #


def generate_term_component_map(ate_term_base_scores, max_n_in_term, valid_tokens):
    ate_terms_components = {}
    for term in ate_term_base_scores.keys():
        # if "foetus" in term or "pst" in term:
        #     print("")
        norm_parts = normalize_string(term)
        term_ngrams = find_ngrams(norm_parts, max_n_in_term)
        selected_parts = list()
        for term_ngram in term_ngrams:
            # check if this part maps to a phrase that is present in the model
            norm_term_ngram = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                                     term_ngram).strip()  # pattern must keep '_' as word2vec model replaces space with _ in n gram
            if len(norm_term_ngram) > 1 and term_ngram in valid_tokens:
                selected_parts.append(term_ngram)
        ate_terms_components[term] = selected_parts
    return ate_terms_components


def gather_graph_stats(log_file, out_file):
    lines_words_selected=[]
    lines_candidate_hits=[]
    lines_graph_stats=[]
    with open(log_file) as inf:
        for line in inf:
            if "selected out of" in line:
                lines_words_selected.append(line)
            elif "candidate terms contain at least one" in line:
                lines_candidate_hits.append(line)
            elif "OVERALL STATS" in line:
                lines_graph_stats.append(line)

    with open(out_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0,len(lines_graph_stats)):
            values=[]
            #2587 selected out of 3544, percentage=0.7299661399548533
            line_words_selected=lines_words_selected[i]
            tokens=line_words_selected.split(",")[0].split(" ")
            values.append(tokens[0].strip()) #selected
            values.append(tokens[4].strip()) #total

            #4785 out of 5659 candidate terms contain at least one selected word
            line_candidate_hits=lines_candidate_hits[i]
            tokens=line_candidate_hits.split(" ")
            values.append(tokens[0].strip()) #hits
            values.append(tokens[3].strip()) #total candidates

            #COMPLETE 21:30:41, OVERALL STATS: stats: nodes=1492, edges=21249, per init=0, num_of_connected_components=5
            line_graph_stats=lines_graph_stats[i]
            trim=line_graph_stats.index("stats:")+6
            line_graph_stats=line_graph_stats[trim:].strip()
            parts=line_graph_stats.split(",")
            for p in parts:
                v=p.split("=")[1]
                values.append(v)

            csvwriter.writerow(values)


def rank_knowmak_terms(texpr_json_output, knowmak_tsv_file, out_file):
    #select seed terms from knowmak_tsv
    seeds = set()
    with open(knowmak_tsv_file, encoding="utf8") as f:
        lines = f.readlines()
        for l in lines:
            content=l.split("\t")
            seeds.add(content[1])

    #read and rank json
    terms_as_list=list()
    json_data = open(texpr_json_output).read()
    data = json.loads(json_data)

    if type(data) is list:
        for term in data:
            term_str=term['string']
            if term_str in seeds:
                continue
            terms_as_list.append((term['string'], term['score-mult']))
    else:
        for k,v in data.items():
            if k in seeds:
                continue
            terms_as_list.append((k, v))
    terms_as_list.sort(key=lambda x: x[1], reverse=True)

    #saving
    with open(out_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in terms_as_list:
            csvwriter.writerow([item[0],item[1]])


def csv_to_tsv(in_file, out_file):
    with open(in_file, newline='') as csvfile:
         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         with open(out_file, 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                csvwriter.writerow(row)


csv_to_tsv("/home/zqz/Work/data/texpr/texpr_output/knowmak/small-RANKED_window=100,top100,sim=0.4,ate=PU.csv",
            "/home/zqz/Work/data/texpr/texpr_output/knowmak/small-RANKED_window=100,top100,sim=0.4,ate=PU.tsv")
# csv_to_tsv("/home/zqz/Work/data/texpr/word_weights/knowmak/100.txt",
#            "/home/zqz/Work/data/texpr/word_weights/knowmak/100.tsv")


# rank_knowmak_terms("/home/zqz/Work/data/texpr/texpr_output/knowmak/RANKED_window=100,top100,sim=0.4,ate=PU.csv",
#                    "/home/zqz/GDrive/project/texpr/data/mostSim4Onto/try4/mostSim4Onto-glove.840B-sim-99.tsv",
#                    "/home/zqz/Work/data/texpr/texpr_output/knowmak/RANKED_window=100,top100,sim=0.4,ate=PU.csv")
#gather_graph_stats("/home/zqz/Work/texpr/genia_.log","/home/zqz/Work/texpr/genia_stats.csv")

#IN_CORPUS="/home/zqz/GDrive/papers/cicling2017/data/semrerank/corpus/genia.tar.gz"
#genia_corpus_to_unigrams(IN_CORPUS, "/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min2_per_file_unigram")
# ate_term_base_scores = {c[0]: c[1] for c in jate_terms_iterator("/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/min1/texpr_base.json")}
#
#
