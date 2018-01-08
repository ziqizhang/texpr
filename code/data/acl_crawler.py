import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
from textblob import TextBlob

file_output="/home/zqz/Work/data/texpr/dict/acl_anthology.txt"
webpages=["http://aclweb.org/anthology/P/P17","http://aclweb.org/anthology/P/P16",
          "http://aclweb.org/anthology/P/P15","http://aclweb.org/anthology/P/P14",
          "http://aclweb.org/anthology/P/P13","http://aclweb.org/anthology/P/P12",
          "http://aclweb.org/anthology/P/P11","http://aclweb.org/anthology/P/P10",
          "http://aclweb.org/anthology/P/P09","http://aclweb.org/anthology/P/P08",
          "http://aclweb.org/anthology/P/P07","http://aclweb.org/anthology/P/P06",
          "http://aclweb.org/anthology/P/P05","http://aclweb.org/anthology/P/P04",
          "http://aclweb.org/anthology/P/P03","http://aclweb.org/anthology/P/P02",
          "http://aclweb.org/anthology/P/P01","http://aclweb.org/anthology/P/P00",
          "http://aclweb.org/anthology/E/E17","http://aclweb.org/anthology/E/E14",
          "http://aclweb.org/anthology/E/E12","http://aclweb.org/anthology/E/E09",
          "http://aclweb.org/anthology/E/E06","http://aclweb.org/anthology/E/E03",
          "http://aclweb.org/anthology/N/N16","http://aclweb.org/anthology/N/N15",
          "http://aclweb.org/anthology/N/N13","http://aclweb.org/anthology/N/N12",
          "http://aclweb.org/anthology/N/N16","http://aclweb.org/anthology/N/N10",
          "http://aclweb.org/anthology/N/N09","http://aclweb.org/anthology/N/N07",
          "http://aclweb.org/anthology/N/N06","http://aclweb.org/anthology/N/N04",
          "http://aclweb.org/anthology/N/N03","http://aclweb.org/anthology/N/N01",
          "http://aclweb.org/anthology/N/N00"]

regex = re.compile('[^a-zA-Z0-9]')
candidates={}
for url in webpages:
    print(url)
    response = urlopen(url)
    soup = BeautifulSoup(response.read(), 'html.parser')
    papers = soup.find_all('i')
    print("\t{}".format(len(papers)))
    i=0
    for p in papers:
        if i==0:
            i+=1
            continue
        title=p.text
        blob = TextBlob(title)
        phrases=blob.noun_phrases
        for p in phrases:
            #First parameter is the replacement, second parameter is your input string
            p=regex.sub(' ', p)
            p=p.strip()
            if p in candidates.keys():
                candidates[p]=candidates[p]+1
            else:
                candidates[p]=1
        i+=1
        if i%50==0:
            print("\t\textracted {}".format(len(candidates)))

sorted=list()
min_freq=2
min_tok=2
for k,v in candidates.items():
    if v>=min_freq and len(k.split(" "))>=min_tok:
        sorted.append(k)
sorted.sort()
print("total after filter {}".format(len(sorted)))
with open(file_output, 'w') as the_file:
    for e in sorted:
        the_file.write(e+'\n')



