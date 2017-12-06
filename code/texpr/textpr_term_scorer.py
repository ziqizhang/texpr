import datetime
import json
import os
import re

from nltk.corpus import stopwords
from textrank.texpr import  utils

from texpr import semrerank_scorer as scorer


#ate_ref_candidate_terms_list - a list of candidate terms, extracted by an ATE api.
#ate_ranked_terms_per_algorithm_folder - a folder containing the list of ATE extraction results by a set of algorithms
def run_textpr(ate_ref_candidate_terms_list, stopwords,
               ate_ranked_terms_per_algorithm_folder,
               textpr_score_file,
               out_folder,
               append_label=None): #append_label: a suffix to be attached to every output file
    textpr_scores=load_textrank_scores(textpr_score_file)
    compute(textpr_scores,
            ate_ref_candidate_terms_list,
            stopwords, ate_ranked_terms_per_algorithm_folder, out_folder,
            append_label)


def run_kcrdc_baseline(word2vec_model, jate_terms_file, stopwords, jate_terms_folder, kcr_score_file, out_folder):
    kcr_lookup=load_kcr_scores(kcr_score_file)
    compute(kcr_lookup, word2vec_model, jate_terms_file,
            stopwords, jate_terms_folder, out_folder)


def load_textrank_scores(in_file):
    score_lookup={}
    with open(in_file, encoding="utf-8") as file:
        kcr_words = file.readlines()
        for item in kcr_words:
            splits = item.split(",")
            score_lookup[splits[0]]=float(splits[1])
    return score_lookup


def load_kcr_scores(in_file):
    score_lookup={}
    with open(in_file, encoding="utf-8") as file:
        kcr_words = file.readlines()
        for item in kcr_words:
            start=item.index('(')+1
            end=item.index('[')
            word = item[start:end].strip()

            score_start=item.rfind(',')+1
            score_end=len(item)-2
            score=float(item[score_start: score_end])
            score_lookup[word]=score
    return score_lookup


def compute(textpr_scores, ate_ref_candidate_terms_for_corpus, stopwords,
            folder_of_ate_ranked_terms_by_alg, out_folder, append_label=None):
    ate_term_base_scores = {c[0]: c[1] for c in utils.jate_terms_iterator(ate_ref_candidate_terms_for_corpus)}
    unigrams_from_ate_terms = set()
    for term in ate_term_base_scores.keys():
        norm_parts = utils.normalize_string(term)
        for part in norm_parts:
            part = re.sub(r'[^a-zA-Z0-9,/\-\+\s_]', ' ',
                          part).strip()
            if (part in stopwords or len(part) < 2):
                continue
            else:
                unigrams_from_ate_terms.add(part)

    #container for term composite unigram and its textpr calculated weights
    sum_unigram_textpr_weights = {}
    for tu in unigrams_from_ate_terms:
        if tu in textpr_scores.keys():
            sum_unigram_textpr_weights[tu]=textpr_scores[tu]
        else:
            sum_unigram_textpr_weights[tu]=0.0
    sum_unigram_textpr_weights = utils.normalize(sum_unigram_textpr_weights)

    #calculate for each ate term candidate, its component n-grams
    #todo: check what are the textpr_scores.keys() are they stemmed, or lemmatised, or unprocessed
    ate_terms_components = utils.generate_term_component_map(ate_term_base_scores,
                                                             5, textpr_scores.keys())

    for file in os.listdir(folder_of_ate_ranked_terms_by_alg):
        print("\t{}".format(file))
        ate_term_base_scores = {c[0]: c[1] for c in utils.jate_terms_iterator(
            folder_of_ate_ranked_terms_by_alg + "/" + file)}
        modified_scores = scorer.SemReRankScorer(sum_unigram_textpr_weights, ate_terms_components,
                                         ate_term_base_scores)
        out_file=out_folder+"/"+file
        if append_label is not None:
            out_file=out_file+"_"+append_label
        # sorted_term_rank_scores = sorted(list(term_rank_scores), key=lambda k: k['score'])
        with open(out_file, 'w') as outfile:
            json.dump(list(modified_scores), outfile)



# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/ttcw_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/ttcw"

# ATE_ALG_SET="" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1/cvalue.json"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/ttcm_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/ttcm"

# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/acl_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/aclrd_ver2"

# ATE_ALG_SET="" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia"+ATE_ALG_SET+"/min1/cvalue.json"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# word_weight_file= "//home/zqz/Work/atr4s/experiments/output/genia_dc_word/PostRankDC.txt"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/dc/genia"
# run_kcrdc_baseline(embedding_model, jate_terms_file, stop, jate_terms_folder, word_weight_file, out_folder)
#





# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcm-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# #word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_ttcm.txt"
# word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/ttcm/atr4s"
# #out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/ttcm"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank_per_unsup/ttcm"

# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_ttcw-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1/Basic.txt"
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind"+ATE_ALG_SET+"/min1"
# stop = stopwords.words('english')
# #word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_ttcw.txt"
# word_weight_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/ttcw/atr4s"
# #out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/ttcw"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank_per_unsup/ttcw"

# ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
# embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_aclv2-uni-sg-100-w3-m1.model"
# jate_terms_file="/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1/Basic.txt"
# stop = stopwords.words('english')
# jate_terms_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2"+ATE_ALG_SET+"/min1"
# #word_weight_file= "/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_aclv2.txt"
# word_weight_file= "/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/aclv2/atr4s"
# #out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/aclrd_ver2"
# out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank_per_unsup/"

ATE_ALG_SET="_atr4s" #use empty string for jate ate algorithms' output; _atr4s for atr4s' algorithms
embedding_model = "/home/zqz/Work/data/semrerank/embeddings/em_g-uni-sg-100-w3-m1.model"
ate_ref_candidate_list= "/home/zqz/Work/data/semrerank/jate_lrec2016/genia" + ATE_ALG_SET + "/min1/Basic.txt"
stop = stopwords.words('english')
ate_ranked_terms_per_algorithm_folder = "/home/zqz/Work/data/semrerank/jate_lrec2016/genia" + ATE_ALG_SET + "/min1"
#word_weight_file= "/home/zqz/Work/data/semrerank/word_weights/textrank/v2/words_genia.txt"
word_weight_file= "/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/genia/atr4s"
#out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank/genia"
out_folder="/home/zqz/Work/data/semrerank/ate_output/textrank_per_unsup"


if word_weight_file.endswith(".txt"):
    run_textpr(ate_ref_candidate_list, stop,
               ate_ranked_terms_per_algorithm_folder,
               word_weight_file,
               out_folder)
else:
    for file in os.listdir(word_weight_file):
        if "words_" not in file:
            continue
        print(">> Word weight file: {}, time: {}".format(file,str(datetime.datetime.now())))
        label_index=file.rfind('.')
        append_label=file[label_index:]
        run_textpr(ate_ref_candidate_list, stop,
                   ate_ranked_terms_per_algorithm_folder,
                   word_weight_file +"/" + file,
                   out_folder, append_label)
        #print("\n")
