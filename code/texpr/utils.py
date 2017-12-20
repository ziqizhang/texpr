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
    for term in data:
        count = count + 1
        yield term['string'], term['score']
        # if (count % 2000 == 0):
        #     logger.info("\t loaded {}".format(count))
        #


def generate_term_component_map(ate_term_base_scores, max_n_in_term, valid_tokens):
    ate_terms_components = {}
    for term in ate_term_base_scores.keys():
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


IN_CORPUS="/home/zqz/GDrive/papers/cicling2017/data/semrerank/corpus/genia.tar.gz"
#genia_corpus_to_unigrams(IN_CORPUS, "/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min2_per_file_unigram")




