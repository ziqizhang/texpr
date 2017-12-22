import json
import logging
import os
import re
import sys

import networkx as nx
import datetime
import time
from textrank.summa import utils

from nltk.corpus import stopwords

from textrank.summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from textrank.summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from textrank.summa.commons import add_graph_nodes
from textrank.summa import keywords as ky
from texpr import texpr_term_scorer as ts

LEMMATIZE_OR_STEM = 1  # 0-lemmatize all vertices; 1-stem all vertices
MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT = 0.5
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/home/zqz/Work/data/semrerank/log/textrank.log', level=logging.INFO, filemode='w')
WINDOW_SIZE = 10

"""Check tags in http://www.clips.ua.ac.be/pages/mbsp-tags and use only first two letters
Example: filter for nouns and adjectives:
INCLUDING_FILTER = ['NN', 'JJ']"""
# INCLUDING_FILTER = ['NN', 'JJ','NNS','NNP','IN','VBN','VBG','CC','TO','VBP','VBN','VBZ','CD']
INCLUDING_FILTER = ['NN', 'JJ']
EXCLUDING_FILTER = []
stop = stopwords.words('english')
min_char = 3


# the method will output textpr weights for every word, UNNORMALIZED form.
def filter_tokens(tokens, filters):
    toks={}
    if filters is not None and len(filters)>0:
        for key, val in tokens.items():
            if key in filters:
                toks[key]=val
        return toks
    return tokens


def keywords_to_ate_percorpus(in_folder, out_file, window_size,num_of_personalized=None, sorted_seed_terms=None,
                              gs_term_file=None, filters=None):
    count = 0
    total_non_zero_elements_pnl_init = 0
    graph = nx.Graph()
    edge_weights = {}
    all_tokens = {}
    for file in os.listdir(in_folder):
        count += 1
        #print("\t"+str(count) + "," + "," + str(datetime.datetime.now()) + "," + file)
        with open(in_folder + '/' + file, 'r') as myfile:
            text = myfile.read()

            # Gets a dict of word -> lemma
            tokens = _clean_text_by_word(text, "english")
            tokens=filter_tokens(tokens, filters)
            split_text = list(_tokenize_by_word(text))

            # Creates the graph and adds the edges
            graph = add_graph_nodes(ky._get_words_for_graph(tokens), graph)
            ky._update_graph_edges(graph, tokens, split_text, edge_weights,window_size)
            del split_text  # It's no longer used
            all_tokens.update(tokens)

    # logger.info("{}: {} graph stats: nodes={}, edges={}".format(
    #     time.strftime("%H:%M:%S"), file, len(graph.nodes()), len(graph.edges())))


    ###############
    # after all edge weights updated, now create edges on graph
    ###############
    ky._set_graph_edges_weighted(graph, edge_weights)
    print("\t{}: graph stats before personalization: nodes={}, edges={}".format(
        time.strftime("%H:%M:%S"), len(graph.nodes()), len(graph.edges())))
    personalized_init = None
    non_zero_elements_pnl_init = 0
    if num_of_personalized is not None:
        output = \
            init_personalized_vector(graph.nodes(), sorted_seed_terms, num_of_personalized,
                                     MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT,
                                     gs_term_file)
        personalized_init = output[0]
        non_zero_elements_pnl_init = output[1]
        total_non_zero_elements_pnl_init = non_zero_elements_pnl_init

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    print("\trunning pagerank...{}".format(datetime.datetime.now()))
    pagerank_scores = nx.pagerank(graph,
                                  alpha=0.85, personalization=personalized_init,
                                  max_iter=5000, tol=1e-06)

    extracted_lemmas = ky._extract_tokens(graph.nodes(), pagerank_scores, 1.0, words=None)
    lemmas_to_word = ky._lemmas_to_words(all_tokens)
    # todo: may apply lemmatizer here to use lemma for key, to be consistent with jate output
    keywords_textpr_weights = ky._get_keywords_with_score(extracted_lemmas, lemmas_to_word)

    # if combined:
    #     combined_keywords = ky._get_combined_keywords(keywords, text.split())
    #     combined_keywords_scores = ky._format_results(keywords, combined_keywords, False, True)
    #     for item in combined_keywords_scores:
    #          if item[0] in textrank_scores.keys():
    #             score = textrank_scores[item[0]]
    #             score += item[1]
    #             textrank_scores[item[0]] = score
    #          else:
    #                textrank_scores[item[0]] = item[1]
    # else:

    print("\t{}: graph stats: nodes={}, edges={}, per init={}".format(
        time.strftime("%H:%M:%S"), len(graph.nodes()), len(graph.edges()),
        non_zero_elements_pnl_init))
    if num_of_personalized is not None:
        out_file += "_"+str(num_of_personalized)
    print("\tCOMPLETE {}, OVERALL STATS: {} graph stats: nodes={}, edges={}, per init={}".format(
        time.strftime("%H:%M:%S"), out_file, len(graph.nodes), len(graph.edges),
        total_non_zero_elements_pnl_init))
    f = open(out_file, 'w')
    for key, value in keywords_textpr_weights.items():
        trimmed = key[0:len(key)]
        try:
            f.write(trimmed + "," + str(value) + "\n")  # python will convert \n to os.linesep
        except ValueError:
            pass
    f.close()
    print("end")
    # sys.exit(0)


def init_personalized_vector(graph_nodes, sorted_jate_terms, topN, max_percentage, gs_term_file=None):
    selected = 0
    initialized = set()
    init_vector = dict([(key, 0.0) for key in graph_nodes])

    max_init_vertices = max_percentage * len(graph_nodes)

    if (gs_term_file is None):
        gs_terms_list = []
    else:
        gs_terms_list = utils.read_and_normalize_terms(gs_term_file, LEMMATIZE_OR_STEM)
        # print("supervised graph")

    for key in sorted_jate_terms:
        selected = selected + 1

        key = utils.lemmatizer.lemmatize(key).strip().lower()
        key = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', key).strip()
        if len(gs_terms_list) > 0 and len(key) > 2:
            if key not in gs_terms_list:
                continue

        for unigram in utils.normalize_string(key, LEMMATIZE_OR_STEM):
            if len(re.sub('[^0-9a-zA-Z]+', '', unigram)) < 2:
                continue
            if unigram in graph_nodes:
                init_vector[unigram] = 1.0
                initialized.add(unigram)
        if selected >= topN:
            # logger.info("personalization init non zero elements:{}, from top {}, with max initialisable vertices {}"
            #       .format(initialized, selected, max_init_vertices))
            # print("personalization init non zero elements:{}, from top {}, with max initialisable vertices {}"
            #       .format(initialized, selected, max_init_vertices))
            break

    if len(initialized) == 0:
        return [None, 0]
    return [init_vector, len(initialized)]


def select_words_as_nodes_fromjson(sim_scores_folder: str, topn:float, min_sim=0.0):
    selected = set()
    all = set()
    for file in os.listdir(sim_scores_folder):
        with open(sim_scores_folder+"/"+file, encoding='utf8') as json_data:
            data = json.load(json_data)
            print("\t processing... {}, items={}".format(file,len(data)))
            count=0
            for key, value in data.items():
                count+=1
                #value.sort(key=lambda x: x[1], reverse=True)
                if topn<1.0:
                    cutoff_index = int(len(value) * topn)
                else:
                    cutoff_index=int(topn)
                for i in range(0, cutoff_index):
                    if float(value[i][1])>min_sim:
                        selected.add(value[i][0])
                    all.add(value[i][0])
                for i in range(cutoff_index, len(value)):
                    all.add(value[i][0])
                if count%50==0:
                    print("\t\t "+str(count))
    print("\t{} selected out of {}".format(str(len(selected)),str(len(all))))
    return selected


# personalized = [50, 100, 200]
# topn_for_graph_nodes = [0.1, 0.2, 0.5]
#
# ATE_LIBRARY = "atr4s"
# in_folder = "/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/raw_abstract_plain_txt"
# out_file = "/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/aclv2/" + ATE_LIBRARY + "/words_aclv2.txt"
# if ATE_LIBRARY == "atr4s":
#     personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf.json"
# else:
#     personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json"
# gs_terms_file = None  # "/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms.txt"
# gs_term_sim_file = "/home/zqz/Work/data/texpr/sim_genia.json"

# ATE_LIBRARY= "atr4s"
# in_folder="/home/zqz/Work/data/jate_data/genia_gs/text/files_standard"
# out_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/genia/"+ATE_LIBRARY+"/words_genia.txt"
# if ATE_LIBRARY=="atr4s":
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/ttf.json"
# else:
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
# gs_terms_file=None#"/home/zqz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2.txt"


# ATE_LIBRARY= "atr4s"
# in_folder="/home/zqz/Work/data/jate_data/ttc/en-mobile-technology/txt_utf8"
# out_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/ttcm/"+ATE_LIBRARY+"/words_ttcm.txt"
# gs_terms_file=None#"/home/zqz/Work/data/jate_data/ttc/gs-en-mobile-technology.txt"
# if ATE_LIBRARY=="atr4s":
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile_atr4s/ttf.json"
# else:
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_mobile/ttf.json"


# ATE_LIBRARY= "atr4s"
# in_folder="/home/zqz/Work/data/jate_data/ttc/en-windenergy/txt_utf8"
# out_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/ttcw/"+ATE_LIBRARY+"/words_ttcw.txt"
# gs_terms_file=None#"/home/zqz/Work/data/jate_data/ttc/gs-en-windenergy.txt"
# if ATE_LIBRARY=="atr4s":
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind_atr4s/ttf.json"
# else:
#     personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/ttc_wind/ttf.json"

#print(len(INCLUDING_FILTER))
def create_setting_label(params):
    label="filter_by_sim="+params["filter_by_sim"]\
          +"-window="+params["window"]+"-ate_alg=0"
    if params["filter_by_sim"]=="True":
        label+="-topnsim="+params["topn"]
        if "min_sim" in params:
            label+="-min_sim="+params["min_sim"]
    return label

sys_argv=sys.argv
if len(sys.argv)==2:
    sys_argv= sys.argv[1].split(" ")

params={}
for arg in sys_argv:
    pv=arg.split("=",1)
    if(len(pv)==1):
        continue
    params[pv[0]]=pv[1]
setting_label=create_setting_label(params)

print(setting_label)
# textrank
if params["filter_by_sim"]=="True": #params["sim_score_files"].endswith(".json"):
    if "min_sim" in params:
        mins=float(params["min_sim"])
    else:
        mins=0.0
    print("Selecting top {} similar words as graph nodes. {}".format(params["topn"],datetime.datetime.now()))
    selected_domain_similar_words = \
        select_words_as_nodes_fromjson(params["sim_score_files"], float(params["topn"]), mins)
else:
    print("No filtering over words to be selected as graph nodes")
    selected_domain_similar_words=None

word_rankscore_folder=params["sys_folder"]
if not os.path.exists(word_rankscore_folder):
    os.makedirs(word_rankscore_folder)
if params["filter_by_sim"]=="True":
    word_rankscore_folder+="/"+params["topn"]+".txt"
else:
    word_rankscore_folder+="/all.txt"


sorted_seed_terms = None
pr_seed_num=None
gs_file=None
if "pr_seed" in params.keys() and params["pr_seed"] is not None:
    print("Using personalized pagerank, pr_seed= {}".format(params["pr_seed"],datetime.datetime.now()))
    pr_jate_term_ttf= {c[0]: c[1] for c in utils.jate_terms_iterator(params["pr_seed"])}
    sorted_seed_terms = sorted(pr_jate_term_ttf, key=pr_jate_term_ttf.get, reverse=True)
if "pr_seed_num" in params.keys():
    pr_seed_num=int(params["pr_seed_num"])
if "gs_file" in params.keys():
    gs_file=int(params["gs_file"])

print("\n>>> SETTING={}".format(setting_label))
print("Computing corpus-level textrank scores. {}".format(datetime.datetime.now()))
keywords_to_ate_percorpus(params["in_corpus"], word_rankscore_folder,
                          int(params["window"]),
                          num_of_personalized=pr_seed_num, sorted_seed_terms=sorted_seed_terms,
                          gs_term_file=gs_file, filters=selected_domain_similar_words)  # personalized textrank

print("Computing final term scores. {}".format(datetime.datetime.now()))
#use_ate_pre_computed=params["ate_alg"] #0 means use pre-computed ate output, from a folder; 1 means
#use average similarity score for a word as base score
#if use_ate_pre_computed=="1":
#    print("\t generate candidate terms and their scores using average sem-sim...")
#    #todo

out_folder=params["outfolder"]+"/"+setting_label
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
ts.run_textpr(params["ate_terms_outfile"], stopwords.words('english'),
               params["ate_terms_outfolder"],
               word_rankscore_folder,
               out_folder,setting_label)


#
# filter_by_sim=False
# window=5
# topn=100
# sim_score_files=/home/zqz/Work/data/texpr/acl_sim/with_dict
# sys_folder=/home/zqz/Work/data/texpr/word_weights/acl
# ate_alg=0
# ate_terms_outfile=/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json
# ate_terms_outfolder=/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/min1
# outfolder=/home/zqz/Work/data/texpr/texpr_output/acl
# in_corpus=/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/raw_abstract_plain_txt

#
# filter_by_sim=False
# window=100
# topn=100
# sim_score_files=/home/zqz/Work/data/texpr/genia_sim/with_dict
# sys_folder=/home/zqz/Work/data/texpr/word_weights/genia
# ate_alg=0
# ate_terms_outfile=/home/zqz/Work/data/semrerank/jate_lrec2016/genia/ttf.json
# ate_terms_outfolder=/home/zqz/Work/data/semrerank/jate_lrec2016/genia/min1
# outfolder=/home/zqz/Work/data/texpr/texpr_output/genia
# in_corpus=/home/zqz/Work/data/jate_data/genia_gs/text/files_standard
