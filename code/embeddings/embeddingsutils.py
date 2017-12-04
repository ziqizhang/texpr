import sys
import gensim
from gensim.models.word2vec import Vocab
import numpy as np
import nltk
import re
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
        # self.ignorepattern=re.compile("[!,-;?£%:]+|[0-9]+")
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

    # returns a tuple where the first element is the list of words found in
    # the embeddings model, and the second is a list of words not found.
    def knownWords(self,words):
        found = []
        notfound = []
        for word in words:
            if word in self.model.vocab:
                found.append(word)
            else:
                notfound.append(word)
        return (found,notfound)

    # return a list of filtered/cleaned/transformed words ready to be used for similarity calculation
    def words4text(self,text):
        tmpwords = nltk.word_tokenize(text)
        if not self.isCaseSensitive:
            tmpwords = [word.lower() for word in tmpwords]
        if self.debug: print("After optional lowercasing: ",tmpwords,"from input",input,file=sys.stderr)
        tmpwords = [word for word in tmpwords if not self.ignorepattern.match(word)]
        if self.filterStopWords:
            tmpwords = [word for word in tmpwords if word.lower() not in self.stopWords]
        words = []
        for word in tmpwords:
            if word in self.model.vocab:
                words.append(word)
            else:
                if self.isCaseSensitive and self.fallBackToLower:
                    if word.lower() in self.model.vocab:
                        words.append(word.lower())
                    else:
                        self.missingWords[word.lower()]=1
                else:
                    self.missingWords[word]=1
        return words
    # return triple of similarity, text1 used, text2 used
    def sim4texts(self,text1,text2):
        words1 = self.words4text(text1)
        words2 = self.words4text(text2)
        if self.debug: print("Words1:",words1,"from input",text1,file=sys.stderr)
        if self.debug: print("Words2:",words2,"from input",text2,file=sys.stderr)
        (sim,used1,used2) = self.sim4words(words1,words2)
        return (sim," ".join(used1)," ".join(used2))

    # return triple of similarity, list1 used, list2 used
    def sim4words(self,words1,words2):
        if len(words1) > 0 and len(words2) > 0:
            ## actually calculate the similarity between the two lists of words
            ## All words in words1 and words should be in the model so we simply
            ## look up the corresponding embeddings
            embs1 = np.array([self.model[w] for w in words1])
            embs2 = np.array([self.model[w] for w in words2])
            ## calculate the average vector for embs1 and embs2
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
    # second is the text actually used for the comparison
    # This uses mostSimilar4Words internally
    def mostSimilar4Text(self,text,n):
        words = self.words4text(text)
        (matches,words_used) = self.mostSimilar4Words(words,n)
        return (matches," ".join(words_used))

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
