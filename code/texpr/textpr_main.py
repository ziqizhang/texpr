import json
import logging
import os
import re

import networkx as nx
import datetime
import time
from textrank.summa import utils

from nltk.corpus import stopwords

from textrank.summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from textrank.summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from textrank.summa.commons import add_graph_nodes
from textrank.summa import keywords as ky

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


#the method will output textpr weights for every word, UNNORMALIZED form.
def keywords_to_ate_percorpus(in_folder, out_file, num_of_personalized=None, sorted_seed_terms=None,
                           gs_term_file=None, filter_tokens=None):
    count = 0
    total_non_zero_elements_pnl_init = 0
    graph=nx.Graph()
    edge_weights={}
    all_tokens={}
    for file in os.listdir(in_folder):
        count += 1
        print(str(count) + "," + "," + str(datetime.datetime.now()) + "," + file)
        with open(in_folder + '/' + file, 'r') as myfile:
            text = myfile.read()

            # Gets a dict of word -> lemma
            tokens = _clean_text_by_word(text, "english")
            #todo: filter tokens
            split_text = list(_tokenize_by_word(text))

            # Creates the graph and adds the edges
            graph = add_graph_nodes(ky._get_words_for_graph(tokens),graph)
            ky._update_graph_edges(graph, tokens, split_text,edge_weights)
            del split_text  # It's no longer used
            all_tokens.update(tokens)

    # logger.info("{}: {} graph stats: nodes={}, edges={}".format(
    #     time.strftime("%H:%M:%S"), file, len(graph.nodes()), len(graph.edges())))


    ###############
    #after all edge weights updated, now create edges on graph
    ###############
    ky._set_graph_edges_weighted(graph,edge_weights)
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
    pagerank_scores = nx.pagerank(graph,
                                    alpha=0.85, personalization=personalized_init,
                                     max_iter=5000, tol=1e-06)

    extracted_lemmas = ky._extract_tokens(graph.nodes(), pagerank_scores, 1.0, words=None)
    lemmas_to_word = ky._lemmas_to_words(all_tokens)
    #todo: may apply lemmatizer here to use lemma for key, to be consistent with jate output
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

    logger.info("\t{}: graph stats: nodes={}, edges={}, per init={}".format(
            time.strftime("%H:%M:%S"), len(graph.nodes()), len(graph.edges()),
            non_zero_elements_pnl_init))


    out_file += str(num_of_personalized)
    logger.info("\n COMPLETE {}, OVERALL STATS: {} graph stats: nodes={}, edges={}, per init={}".format(
            time.strftime("%H:%M:%S"), out_file, len(graph.nodes), len(graph.edges),
            total_non_zero_elements_pnl_init))
    f = open(out_file, 'w')
    for key, value in keywords_textpr_weights.iteritems():
        trimmed = key[0:len(key)]
        try:
            f.write(trimmed + "," + str(value) + "\n")  # python will convert \n to os.linesep
        except ValueError:
            pass
    f.close()
    print ("end")
    #sys.exit(0)


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


def select_words_as_nodes(sim_file:str, topn_percent):
    selected=set()
    all=set()
    with open(sim_file, encoding='utf8') as json_data:
        data = json.load(json_data)
        for key, value in data.items():
            value.sort(key=lambda x: x[1], reverse=True)
            cutoff_index=int(len(value)*topn_percent)
            for i in range(0,cutoff_index):
                selected.add(value[i][0])
                all.add(value[i][0])
            for i in range(cutoff_index,len(value)):
                all.add(value[i][0])
    return selected


personalized = [50, 100, 200]
topn_for_graph_nodes=[0.1,0.2,0.5]

ATE_LIBRARY="atr4s"
in_folder = "/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/raw_abstract_plain_txt"
out_file = "/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/aclv2/"+ATE_LIBRARY+"/words_aclv2.txt"
if ATE_LIBRARY=="atr4s":
    personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf.json"
else:
    personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json"
gs_terms_file = None#"/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms.txt"
gs_term_sim_file="/home/zqz/Work/data/texpr/sim_genia.json"

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

print(len(INCLUDING_FILTER))


# textrank
# keywords_to_ate(in_folder,out_file, False,
#                 num_of_personalized=None, sorted_seed_terms=None,
#                     gs_term_file=None)

# personalized textrank
jate_term_ttf = {c[0]: c[1] for c in utils.jate_terms_iterator(personalization_seed_term_file)}
sorted_seed_terms = sorted(jate_term_ttf, key=jate_term_ttf.get, reverse=True)
for num_personalized_nodes in personalized:
    for topn in topn_for_graph_nodes:
        selected_words = select_words_as_nodes(gs_term_sim_file, topn)
        print('Personalized textrank, {} nodes to be personalized'.format(num_personalized_nodes))
        keywords_to_ate_percorpus(in_folder, out_file,
                               num_of_personalized=num_personalized_nodes, sorted_seed_terms=sorted_seed_terms,
                               gs_term_file=gs_terms_file, filter_tokens=selected_words)