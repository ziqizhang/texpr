## Author: Johann Petrak <johann.petrak@gmail.com>


import sys
import gensim
from gensim.models.word2vec import Vocab, Word2Vec
import numpy as np
import nltk
import re
import math
import itertools
from nltk.corpus import stopwords
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
            level=logging.INFO)

class EmbeddingsUtils:
    def __init__(self):
        self.model = None  ## we will store the model there
        self.verbose = False
        self.debug = False
        self.isCaseSensitive = False
        self.fallBackToLower = False
        self.filterStopWords = True
        self.stopWords = set(stopwords.words("english"))
        self.missingWords = {}
        self.nEmpty = 0
        self.embFile = ""
        self.default_weight = 0.0
        self.use_weights = False
        self.ignorepattern=re.compile("[—!\"#$%&\\\\'()*+,-/:;<=>?@[\]^_`{|}~£¦]+|[0-9]+")

    # load embeddings
    def loadEmbeddings(self,embFile):
        self.embFile = embFile
        inBinary = ".bin." in embFile
        gensimFormat = ".gensim" in embFile
        if gensimFormat :
            if self.verbose: print("Loading embeddings file from GenSim format", embFile, file=sys.stderr)
            self.model = gensim.models.KeyedVectors.load(embFile,mmap='r')
        else:
            if self.verbose: print("Loading embeddings file from word2vec format file", embFile, "binary: ", inBinary, file=sys.stderr)
            self.model = gensim.models.KeyedVectors.load_word2vec_format(embFile, binary=inBinary, unicode_errors='ignore', encoding='utf8')
            print("Embeddings loaded, words: ",len(self.model.index2word),file=sys.stderr)
            ## if we loaded from gensim format we should already have done that!
            if not gensimFormat:
                print("Pre-computing the L2 norms", file=sys.stderr)
                self.model.init_sims(replace=True)   ## we do not need the non-normalized vectors for now, so save some memory
                print("L2 norms computed",file=sys.stderr)
            else:
                ## we already have the normalized vectors in the syn0 data structure, no need to calculate the syn0norm
                self.model.syn0norm = LowerCase


    def loadWeights(self,filename,wordcol=0,weightcol=1,default_weight="max"):
        """Load mappings from words to weights from a tsv file. This must be
        done after loading the embeddings, and only weights for words found in
        the embeddings are kept. This will respect the case sensitivity setting in
        that if is true, then the weight will be stored for the embedding matching
        case, otherwise, all weights for a lower-cased version of the word will
        be accumulated.
        The default weight can be some value or 'min' or 'max'"""
        if not self.model:
            print("ERROR: cannot load weights before the embeddings have been loaded",file=sys.stderr)
            sys.exit(1)
        self.default_weight = default_weight
        ## we use the model word->idx mapping and store all weights
        ## in a numpy index of the size of the vocabulary
        nan = float('nan')
        self.weights = np.full(len(self.model.vocab),nan)
        minweight = sys.float_info.max
        maxweight = sys.float_info.min
        with open(filename,"rt",encoding="utf-8") as inp:
            for line in inp:
                line = line.rstrip()
                fields = line.split("\t")
                word = fields[wordcol]
                weight = float(fields[weightcol])
                v = self.isInVocabStrict(word)
                if not v and self.fallBackToLower:
                    v = self.isInVocabStrict(word.lower())
                if v:
                    if math.isnan(self.weights[v.index]):
                        self.weights[v.index] = weight
                    else:
                        self.weights[v.index] += weight
                    if weight > maxweight:
                        maxweight = weight
                    if weight < minweight:
                        minweight = weight
        if type(default_weight) == str:
            if default_weight == "min":
                defw = minweight
            elif default_weight == "max":
                defw = maxweight
            else:
                print("ERROR odd default weight: ",default_weight,file=sys.stder)
                sys.exit(1)
        else:
            defw = default_weight
        for i in range(len(self.weights)):
            if math.isnan(self.weights[i]):
                self.weights[i] = defw
        self.default_weight = defw
        self.use_weights = True


    def setWeightsFromEmbeddings(self):
        """set the weights from the word counts stored in the embeddings model
        the weight is set to logs(2)/log(count+1), so it is 1.0 for count 1
        and decreases exponentionally towards zero for increasing counts.
        NOTE: the counts in the model may not be useful and this may not work well!!"""
        if not self.model:
            print("ERROR: cannot load weights before the embeddings have been loaded",file=sys.stderr)
            sys.exit(1)
        self.weights = np.full(len(self.model.vocab),default_weight)
        vocab = self.getVocab()
        l2=np.log(2.0)
        for v in vocab:
            self.weights[v.index] = l2/np.log(v.count+1.0)
        self.use_weights = True

    def getWeight(self,word):
        """return the weight for this word or the default weight"""
        v = self.isInVocabStrict(word)
        if not v and self.fallBackToLower:
            v = self.isInVocabStrict(word.lower())
        if v:
            return self.weights[v.index]
        else:
            return self.default_weight

    def setUseWeights(self,flag):
        self.use_weights = flag

    def setIsCaseSensitive(self,flag):
        self.isCaseSensitive = flag

    def setFallBackToLower(self,flag):
        self.fallBackToLower = flag

    # if stopwords should get filtered, on by default
    def setFilterStopwords(self,flag):
        self.filterStopWords = flag

    def setDebug(self,flag):
        self.debug = flag
        self.verbose = flag

    def setVerbose(self,flag):
        self.verbose = flag

    # reset internal statistics about OOV words
    def initStats(self):
        self.missingWords = {}
        self.nEmpty = 0

    def getVocab(self):
        """Get the model vocabulary, a dictionary from word string to
        vocabulary entry, where each entry contains an index and a count """
        havewv = hasattr(self.model,"wv")
        if havewv:
           return self.model.wv.vocab
        else:
           return self.model.vocab

    def isInVocab(self,word):
        """Checks if a word is in the embeddings vocabulary, respecting case sensitivity and
           fallback to lower case settings."""
        if not self.isCaseSensitive:
           word = word.lower()
        known = self.isInVocabStrict(word)
        if not known and self.isCaseSensitive and self.fallBackToLower:
            known = self.isInVocabStrict(word.lower())
        return known

    def isInVocabStrict(self,word):
        """Checks if the word is in the embeddings vocab as it is (no lowercasing or fallback to
           lowercase done). Returns the vocab entry for true or None for false"""
        return self.getVocab().get(word)

    # returns a tuple where the first element is the list of words found in
    # the embeddings model, and the second is a list of words not found.
    def knownWords(self,words):
        """Checks wich of the words in the list 'words' are contained in the embeddings vocabulary
        and returns a list of two lists: first list contains the known words, secon list the unknown words.
        Settings for case sensitivity and fall back to lower case are respected."""
        found = []
        notfound = []
        for word in words:
            if self.isInVocab(word):
                found.append(word)
            else:
                notfound.append(word)
        return (found,notfound)

    # returns a list of tokens (words) as created by the tokeniser, but
    # not filtered by the embeddings nor lowercased.
    # However, known unwanted tokens are filtered (1 letter tokens) and
    # tokens are cleaned to remove any additional punctuation attached to them.
    def tokens4text(self,text):
        # the NLTK tokeniser leaves slashes and hyphens within words intact,
        # so we first replace those with spaces
        text = re.sub(r"[/—-]"," ",text)
        tmpwords = nltk.word_tokenize(text)
        tmpwords = [word for word in tmpwords if len(word) > 1]
        ## remove any trailing dot or other common punctuation
        ## NOTE: this may be a problem if we have an acronym like A.D.
        ## but for now we ignore that
        tmpwords = [re.sub(r"[.,;:!?]+$","",word) for word in tmpwords]
        tmpwords = [word for word in tmpwords if word and not "." in word]
        return tmpwords

    # return a list of filtered/cleaned/transformed words ready to be used for similarity calculation
    def words4text(self,text):
        """Tokenises the text and returns a list of words, optionally lower cased, stop words filtered"""
        tmpwords = self.tokens4text(text)
        if not self.isCaseSensitive:
            tmpwords = [word.lower() for word in tmpwords]
        if self.debug: print("After optional lowercasing: ",tmpwords,"from input",input,file=sys.stderr)
        tmpwords = [word for word in tmpwords if not self.ignorepattern.match(word)]
        if self.filterStopWords:
            tmpwords = [word for word in tmpwords if word.lower() not in self.stopWords]
        words = []
        for word in tmpwords:
            havewv = hasattr(self.model,"wv")
            if self.isInVocabStrict(word):
                words.append(word)
            else:
                if self.isCaseSensitive and self.fallBackToLower:
                    if self.isInVocabStrict(word.lower()):
                        words.append(word.lower())
                    else:
                        self.missingWords[word.lower()]=1
                else:
                    self.missingWords[word]=1
        return words

    # return triple of similarity, text1 used, text2 used
    def sim4texts(self,text1,text2):
        """compare the two texts and return a triple containing the
        similarity, the words used from the first string and the words used
        from the second string"""
        words1 = self.words4text(text1)
        words2 = self.words4text(text2)
        if self.debug: print("Words1:",words1,"from input",text1,file=sys.stderr)
        if self.debug: print("Words2:",words2,"from input",text2,file=sys.stderr)
        (sim,used1,used2) = self.sim4words(words1,words2)
        return (sim,used1,used2)

    # return triple of similarity, list1 used, list2 used
    def sim4words(self,words1,words2):
        """Calculate the similarity between the words in the two lists. This
           expects the words in the lists to be known to be in the vocabulary and
           also to already be in the correct case.
           If word weights are enabled, the embedding vectors are weighted
           by the corresponding weights for averaging"""
        if len(words1) > 0 and len(words2) > 0:
            ## actually calculate the similarity between the two lists of words
            ## All words in words1 and words should be in the model so we simply
            ## look up the corresponding embeddings
            embs1 = np.array([self.model[w] for w in words1])
            embs2 = np.array([self.model[w] for w in words2])
            ## calculate the average vector for embs1 and embs2
            if self.use_weights:
                weights1 = np.array([self.weights[w] for w in words1])
                weights2 = np.array([self.weights[w] for w in words2])
                mean1 = np.average(embs1,weights=weights1,axis=0)
                mean2 = np.average(embs2,weights=weights2,axis=0)
            else:
                mean1 = np.average(embs1,axis=0)
                mean2 = np.average(embs2,axis=0)
            d = np.dot(mean1,mean2)
            n1 = np.linalg.norm(mean1)
            n2 = np.linalg.norm(mean2)
            sim = d/(n1*n2)
        else:
            sim = 0.0
            result = []
            self.nEmpty = self.nEmpty + 1
        return (sim,words1,words2)

    # calculate n most similar words in the embeddings and return
    # a tuple with two elements: first is the list of tuples (word,sim) and
    # second is the list of tokens actually used for the comparison
    # This uses mostSimilar4Words internally
    def mostSimilar4Text(self,text,n):
        words = self.words4text(text)
        (matches,words_used) = self.mostSimilar4Words(words,n)
        return (matches,words_used)

    # find the n most similar entries from the embeddings model for the
    # list of words given.
    # returns a tuple where the first element is the list of matches, each
    # match being a tuple of word and similarity, and the second element is
    # the words actually used
    def mostSimilar4Words(self,words,n):
        (found,notfound) = self.knownWords(words)
        ## note/todo: decide how to handle the case that no word is found
        if(len(found)==0):
            return ([],[])
        else:
            mostsim = self.model.most_similar(positive=found,topn=n)
            return (mostsim,found)

    def getInfo(self):
        ret = {}
        ret["embFile"]=self.embFile
        ret["embShape"]=self.model.syn0.shape
        ret["missingWords"]=self.missingWords
        ret["nEmpty"]=self.nEmpty
        ret["fallBackToLower"]=self.fallBackToLower
        ret["isCaseSensitive"]=self.isCaseSensitive
        return ret

    def __str__(self):
        return "EmbeddingsUtils:{embfile="+self.embFile+"}"

if __name__ == "__main__":
    eu = EmbeddingsUtils()
    if(len(sys.argv)!=2):
        print("ERROR: need one argument, the embeddings file",file=sys.stderr)
        sys.exit(1)
    eu.loadEmbeddings(sys.argv[1])
    print("Loaded, info is ",eu.getInfo())
    print("Sim between '3D printing technologies' and 'Adaptive optics'",eu.sim4texts("3D printing technologies","Adaptive optics"))
    print("Most similar 10 for text optics",eu.mostSimilar4Text("optics",10))
    print("Most similar 10 for 'adaptive,optics'",eu.mostSimilar4Words(["adaptive","optics"],10))
