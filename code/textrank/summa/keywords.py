import logging
import os
import re
from itertools import combinations as _combinations
from queue import Queue as _Queue

import networkx as nx
import datetime
import time
from nltk.corpus import stopwords

from texpr import utils
from textrank.summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from textrank.summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from textrank.summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from textrank.summa.commons import add_graph_nodes
from textrank.summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes

LEMMATIZE_OR_STEM = 1  # 0-lemmatize all vertices; 1-stem all vertices
MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT = 0.5
logger = logging.getLogger(__name__)
logging.basicConfig(filename='textrank.log', level=logging.INFO, filemode='w')

"""Check tags in http://www.clips.ua.ac.be/pages/mbsp-tags and use only first two letters
Example: filter for nouns and adjectives:
INCLUDING_FILTER = ['NN', 'JJ']"""
# INCLUDING_FILTER = ['NN', 'JJ','NNS','NNP','IN','VBN','VBG','CC','TO','VBP','VBN','VBZ','CD']
INCLUDING_FILTER = ['NN', 'JJ']
EXCLUDING_FILTER = []
stop = stopwords.words('english')
min_char = 3


def _get_pos_filters():
    return frozenset(INCLUDING_FILTER), frozenset(EXCLUDING_FILTER)


def _get_words_for_graph(tokens):
    include_filters, exclude_filters = _get_pos_filters()
    if include_filters and exclude_filters:
        raise ValueError("Can't use both include and exclude filters, should use only one")

    result = []
    for word, unit in tokens.items():
        if exclude_filters and unit.tag in exclude_filters:
            continue
        # if word in stop:
        #     continue
        # if len(word) < min_char:
        #     continue
        if (include_filters and unit.tag in include_filters) or not include_filters or not unit.tag:
            result.append(unit.token)
    return result


def _get_first_window(split_text,window_size):
    return split_text[:window_size]


#todo:tokens: used as a filter to choose only legit words
def _set_graph_edge(graph, tokens, word_a, word_b):
    if word_a in tokens and word_b in tokens:
        lemma_a = tokens[word_a].token
        lemma_b = tokens[word_b].token

        if graph.has_node(lemma_a) and graph.has_node(lemma_b) and not graph.has_edge(lemma_a, lemma_b):
            graph.add_edge(lemma_a, lemma_b)

def _update_graph_edge_weight(graph, tokens, word_a, word_b, edge_weights):
    if word_a in tokens and word_b in tokens:
        lemma_a = tokens[word_a].token
        lemma_b = tokens[word_b].token

        if graph.has_node(lemma_a) and graph.has_node(lemma_b):
            key=','.join(sorted([lemma_a, lemma_b]))
            if graph.has_edge(lemma_a, lemma_b):
                edge_weights[key]=edge_weights[key]+1
            else:
                edge_weights[key]=1


def _process_first_window(graph, tokens, split_text,window_size):
    first_window = _get_first_window(split_text,window_size)
    for word_a, word_b in _combinations(first_window, 2):
        _set_graph_edge(graph, tokens, word_a, word_b)

def _process_first_window_update_edge_weights(graph, tokens, split_text,edge_weights,window_size):
    first_window = _get_first_window(split_text,window_size)
    for word_a, word_b in _combinations(first_window, 2):
        _update_graph_edge_weight(graph, tokens, word_a, word_b,edge_weights)


def _init_queue(split_text,window_size):
    queue = _Queue()
    first_window = _get_first_window(split_text,window_size)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def _process_word(graph, tokens, queue, word):
    for word_to_compare in _queue_iterator(queue):
        _set_graph_edge(graph, tokens, word, word_to_compare)


def _process_word_update_edge_weights(graph, tokens, queue, word,edge_weights):
    for word_to_compare in _queue_iterator(queue):
        _update_graph_edge_weight(graph, tokens, word, word_to_compare,edge_weights)


def _update_queue(queue, word, window_size):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (window_size - 1)


def _process_text(graph, tokens, split_text,window_size):
    queue = _init_queue(split_text,window_size)
    for i in range(window_size, len(split_text)):
        word = split_text[i]
        _process_word(graph, tokens, queue, word)
        _update_queue(queue, word,window_size)


def _process_text_update_edge_weights(graph, tokens, split_text,edge_weights,window_size):
    queue = _init_queue(split_text,window_size)
    for i in range(window_size, len(split_text)):
        word = split_text[i]
        _process_word_update_edge_weights(graph, tokens, queue, word,edge_weights)
        _update_queue(queue, word,window_size)


def _queue_iterator(queue):
    iterations = queue.qsize()
    for i in range(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def _set_graph_edges(graph, tokens, split_text,window_size):
    _process_first_window(graph, tokens, split_text,window_size)
    _process_text(graph, tokens, split_text,window_size)

def _set_graph_edges_weighted(graph,edge_weights):
    for key, value in edge_weights.items():
        nodes=key[0:len(key)].split(",")
        graph.add_edge(nodes[0],nodes[1],weight=value)


def _update_graph_edges(graph, tokens, split_text, edge_weights,window_size):
    _process_first_window_update_edge_weights(graph,tokens,split_text, edge_weights,window_size)
    _process_text_update_edge_weights(graph, tokens,split_text, edge_weights,window_size)

def _extract_tokens(lemmas, scores, ratio, words):
    lemmas=list(lemmas)
    lemmas.sort(key=lambda s: scores[s], reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio, else, the ratio is ignored.
    length = len(lemmas) * ratio if words is None else words
    return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))]


def _lemmas_to_words(tokens):
    lemma_to_word = {}
    for word, unit in tokens.items():
        lemma = unit.token
        if lemma in lemma_to_word:
            lemma_to_word[lemma].append(word)
        else:
            lemma_to_word[lemma] = [word]
    return lemma_to_word


def _get_keywords_with_score(extracted_lemmas, lemma_to_word):
    """
    :param extracted_lemmas:list of tuples
    :param lemma_to_word: dict of {lemma:list of words}
    :return: dict of {keyword:score}
    """
    keywords = {}
    for score, lemma in extracted_lemmas:
        keyword_list = lemma_to_word[lemma]
        for keyword in keyword_list:
            keywords[keyword] = score
    return keywords
    # return {keyword:score for score, lemma in extracted_lemmas for keyword in lemma_to_word[lemma]}
    # if you dare


def _strip_word(word):
    stripped_word_list = list(_tokenize_by_word(word))
    return stripped_word_list[0] if stripped_word_list else ""


def _get_combined_keywords(_keywords, split_text):
    """
    :param keywords:dict of keywords:scores
    :param split_text: list of strings
    :return: combined_keywords:list
    """
    result = []
    _keywords = _keywords.copy()
    len_text = len(split_text)
    for i in range(len_text):
        word = _strip_word(split_text[i])
        if word in _keywords:
            combined_word = [word]
            if i + 1 == len_text: result.append(word)  # appends last word if keyword and doesn't iterate
            for j in range(i + 1, len_text):
                other_word = _strip_word(split_text[j])
                if other_word in _keywords and other_word == split_text[j].decode("utf-8"):
                    combined_word.append(other_word)
                else:
                    try:
                        for keyword in combined_word: _keywords.pop(keyword)
                        result.append(" ".join(combined_word))
                        break
                    except KeyError:
                        pass
    return result


def _get_average_score(concept, _keywords):
    word_list = concept.split()
    word_counter = 0
    total = 0
    for word in word_list:
        total += _keywords[word]
        word_counter += 1
    return total / word_counter


def _format_results(_keywords, combined_keywords, split, scores):
    """
    :param keywords:dict of keywords:scores
    :param combined_keywords:list of word/s
    """
    combined_keywords.sort(key=lambda w: _get_average_score(w, _keywords), reverse=True)
    if scores:
        return [(word, _get_average_score(word, _keywords)) for word in combined_keywords]
    if split:
        return combined_keywords
    return "\n".join(combined_keywords)


def keywords(text, window_size,ratio=0.2, words=None, language="english", split=False, scores=False):
    # Gets a dict of word -> lemma
    tokens = _clean_text_by_word(text, language)
    split_text = list(_tokenize_by_word(text))

    # Creates the graph and adds the edges
    graph=nx.Graph()
    graph = add_graph_nodes(_get_words_for_graph(tokens),graph)
    _set_graph_edges(graph, tokens, split_text,window_size)
    del split_text  # It's no longer used

    _remove_unreachable_nodes(graph)

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)

    extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores, ratio, words)

    lemmas_to_word = _lemmas_to_words(tokens)
    keywords = _get_keywords_with_score(extracted_lemmas, lemmas_to_word)

    # text.split() to keep numbers and punctuation marks, so separeted concepts are not combined
    combined_keywords = _get_combined_keywords(keywords, text.split())

    return _format_results(keywords, combined_keywords, split, scores)


def get_graph(text, window_size,language="english"):
    tokens = _clean_text_by_word(text, language)
    split_text = list(_tokenize_by_word(text))
    graph=nx.Graph()
    graph = add_graph_nodes(_get_words_for_graph(tokens),graph)
    _set_graph_edges(graph, tokens, split_text,window_size)

    return graph


def keywords_to_ate_perdoc(in_folder, out_file, combined, window_size,num_of_personalized=None, sorted_seed_terms=None,
                           gs_term_file=None):
    textrank_scores = {}

    count = 0
    total_non_zero_elements_pnl_init = 0
    total_nodes = 0
    total_edges = 0
    for file in os.listdir(in_folder):
        count += 1
        print(str(count) + "," + "," + str(datetime.datetime.now()) + "," + file)
        with open(in_folder + '/' + file, 'r') as myfile:
            text = myfile.read()

            # Gets a dict of word -> lemma
            tokens = _clean_text_by_word(text, "english")
            split_text = list(_tokenize_by_word(text))

            # Creates the graph and adds the edges
            graph=nx.Graph()
            graph = add_graph_nodes(_get_words_for_graph(tokens),graph)
            _set_graph_edges(graph, tokens, split_text,window_size)
            del split_text  # It's no longer used

            # todo: orignal code has this line
            _remove_unreachable_nodes(graph)

            total_edges += len(graph.edges())
            total_nodes += len(graph.nodes())
            # logger.info("{}: {} graph stats: nodes={}, edges={}".format(
            #     time.strftime("%H:%M:%S"), file, len(graph.nodes()), len(graph.edges())))

            personalized_init = None
            non_zero_elements_pnl_init = 0
            if num_of_personalized is not None:
                output = \
                    init_personalized_vector(graph.nodes(), sorted_seed_terms, num_of_personalized,
                                             MAX_PERCENTAGE_OF_VERTICES_FOR_PERSONALIZATION_INIT,
                                             gs_term_file)
                personalized_init = output[0]
                non_zero_elements_pnl_init = output[1]
                total_non_zero_elements_pnl_init += non_zero_elements_pnl_init

            # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
            pagerank_scores = nx.pagerank(graph,
                                          alpha=0.85, personalization=personalized_init,
                                          max_iter=5000, tol=1e-06)

            extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores, 1.0, words=None)

            lemmas_to_word = _lemmas_to_words(tokens)
            keywords = _get_keywords_with_score(extracted_lemmas, lemmas_to_word)

            if combined:
                combined_keywords = _get_combined_keywords(keywords, text.split())
                combined_keywords_scores = _format_results(keywords, combined_keywords, False, True)
                for item in combined_keywords_scores:
                    if item[0] in textrank_scores.keys():
                        score = textrank_scores[item[0]]
                        score += item[1]
                        textrank_scores[item[0]] = score
                    else:
                        textrank_scores[item[0]] = item[1]
            else:
                for key, value in keywords.iteritems():
                    if key in textrank_scores.keys():
                        score = textrank_scores[key]
                        score += value
                        textrank_scores[key] = score
                    else:
                        textrank_scores[key] = value
        logger.info("\t{}: {} graph stats: nodes={}, edges={}, per init={}".format(
            time.strftime("%H:%M:%S"), file, len(graph.nodes()), len(graph.edges()),
            non_zero_elements_pnl_init))


    out_file += str(num_of_personalized)
    logger.info("\n COMPLETE {}, OVERALL STATS: {} graph stats: nodes={}, edges={}, per init={}".format(
            time.strftime("%H:%M:%S"), out_file, total_nodes, total_edges,
            total_non_zero_elements_pnl_init))
    f = open(out_file, 'w')
    for key, value in textrank_scores.iteritems():
        trimmed = key[0:len(key)]
        try:
            f.write(trimmed + "," + str(value) + "\n")  # python will convert \n to os.linesep
        except ValueError:
            pass
    f.close()
    print("end")
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


personalized = [50, 100, 200]

# ATE_LIBRARY="atr4s"
# in_folder = "/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/raw_abstract_plain_txt"
# out_file = "/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/aclv2/"+ATE_LIBRARY+"/words_aclv2.txt"
# if ATE_LIBRARY=="atr4s":
#     personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2_atr4s/ttf.json"
# else:
#     personalization_seed_term_file = "/home/zqz/Work/data/semrerank/jate_lrec2016/aclrd_ver2/ttf.json"
# gs_terms_file = None#"/home/zqz/Work/data/jate_data/acl-rd-corpus-2.0/acl-rd-ver2-gs-terms.txt"


ATE_LIBRARY= "atr4s"
in_folder="/home/zqz/Work/data/jate_data/genia_gs/text/files_standard"
out_file="/home/zqz/Work/data/semrerank/word_weights/textrank/v2_per_unsup/genia/"+ATE_LIBRARY+"/words_genia.txt"
if ATE_LIBRARY=="atr4s":
    personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia_atr4s/ttf.json"
else:
    personalization_seed_term_file="/home/zqz/Work/data/semrerank/jate_lrec2016/genia/ttf.json"
gs_terms_file=None#"/home/zqz/Work/data/jate_data/genia_gs/concept/genia_gs_terms_v2.txt"


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


# textrank
# keywords_to_ate(in_folder,out_file, False,
#                 num_of_personalized=None, sorted_seed_terms=None,
#                     gs_term_file=None)

# personalized textrank
# jate_term_ttf = {c[0]: c[1] for c in utils.jate_terms_iterator(personalization_seed_term_file)}
# sorted_seed_terms = sorted(jate_term_ttf, key=jate_term_ttf.get, reverse=True)
# for num_personalized_nodes in personalized:
#     print('Personalized textrank, {} nodes to be personalized'.format(num_personalized_nodes))
#     keywords_to_ate_perdoc(in_folder, out_file, False,
#                            num_of_personalized=num_personalized_nodes, sorted_seed_terms=sorted_seed_terms,
#                            gs_term_file=gs_terms_file)
