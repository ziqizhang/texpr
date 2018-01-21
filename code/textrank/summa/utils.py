import json
import re
import codecs

from nltk import WordNetLemmatizer, SnowballStemmer

WORD_EMBEDDING_VOCAB_MIN_LENGTH=3
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("init")#this forces lemmatizer to load, in order to avoid non-thread safe usage of lemmatizer
STEMMER = SnowballStemmer("english")

from nltk.tokenize import word_tokenize
def keep_token(token):
    token = re.sub(r'[^a-zA-Z]', '', token).strip()
    return len(token)>=WORD_EMBEDDING_VOCAB_MIN_LENGTH


def normalize_string(original, lem_or_stem=0):
    toks = word_tokenize(original)
    norm_list = list()
    for tok in toks:
        # if tok=="b_alpha" or 'dihydroxycholecalciferol' in tok:
        #     print("caught")
        if lem_or_stem==0:
            norm = lemmatizer.lemmatize(tok).strip().lower()
        else:
            norm=STEMMER.stem(tok)

        norm = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', norm).strip()
        for part in norm.split():
            if keep_token(norm):
                norm_list.append(part)
    # sent= " ".join(norm_list)
    return norm_list



def read_and_normalize_terms(filePath, lem_or_stem=0):
    list=[]
    if(lem_or_stem==0):
        f= codecs.open(filePath, encoding="utf-8")
        for t in f:
            t = lemmatizer.lemmatize(t.strip()).strip().lower()
            t = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', t).strip()
            list.append(t)
        return list
    else:
        f= codecs.open(filePath, encoding="utf-8")
        for t in f:
            t = STEMMER.stem(t.strip()).strip().lower()
            t = re.sub(r'[^a-zA-Z0-9,/\-\+\s]', ' ', t).strip()
            list.append(t)
        return list


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
