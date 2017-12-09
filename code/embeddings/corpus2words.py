## Author: Johann Petrak <johann.petrak@gmail.com>


from __future__ import print_function
import sys
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from embeddingsutils import EmbeddingsUtils

## simple script to read in files in a directory, tokenise and filter and
## output a list of words and word frequencies for those words which also
## occur in the embeddings file
if(len(sys.argv)!=4):
    print("ERROR: need three arguments, the embeddings file, the corpus directory, the file for saving OOV stats",file=sys.stderr)
    sys.exit(1)

eu = EmbeddingsUtils()
eu.setIsCaseSensitive(False)
eu.setFallBackToLower(False)
eu.setFilterStopwords(True)
eu.setDebug(False)
eu.setVerbose(True)
eu.loadEmbeddings(sys.argv[1])


docs = {}
nOOV = 0
def tokenizer(text):
    global eu
    return eu.words4text(text)

for dirpath, dirs, files in os.walk(sys.argv[2]):
    for f in files:
        file_name = os.path.join(dirpath,f)
        print("Adding document",file_name,file=sys.stderr)
        with open(file_name) as instream:
            text = instream.read()
            docs[file_name] = text

tfidf = TfidfVectorizer(tokenizer=tokenizer)
print("Calculating tfidf ...",file=sys.stderr)
tfs = tfidf.fit_transform(docs.values())
print("DONE, words: ",len(tfidf.idf_),file=sys.stderr)

## tfidf.vocabulary_ is a map word->id
## tfidf.idf_ is an array of idfs
## eu.missingWords is a mpa from word to frequency
## print("OOV words:",eu.missingWords,file=sys.stderr)

print("Number of OOV words:",len(eu.missingWords),file=sys.stderr)
with open(sys.argv[3],"w") as out:
    for word, count in eu.missingWords.items():
        print(word,count,sep="\t",file=out)

for word, wid in tfidf.vocabulary_.items():
    print(word,tfidf.idf_[wid],sep="\t")
