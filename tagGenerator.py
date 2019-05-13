import pandas as pd
import nltk
import string
import numpy as np
import os

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.parse.stanford import StanfordDependencyParser

from gensim.models import word2vec
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stop_words=set(stopwords.words("english"))

parser = StanfordDependencyParser(r"stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar", r"stanford-english-corenlp-2018-10-05-models.jar")

# java path setting

class tagGenerator:

    def __init__(self, feedback):
        self.feedback = feedback
        self.stringList = []
        self.depList = []
        self.model = word2vec.Word2Vec(min_count=2, size=200) # parameter
        self.model_name = 'model/'+self.feedback+'_word2vec'

    # Used for input data cleaning
    def cleanSent(self, sent):
        
        global stop_words
        
        if len(sent) == 0:
            return
        
        res = []
        
        for i in sent:
            trantab = str.maketrans({key: None for key in (string.punctuation + string.digits)}) 
            cleanLine =i.translate(trantab)
            cleanLine = cleanLine.lower() 
            cleanWord = word_tokenize(cleanLine) # word toeknize
            tmp_res = []

            for word in cleanWord:
                if word not in stop_words:
                    if wordnet.morphy(word):
                        tmp_res.append(wordnet.morphy(word)) # stemming
                    else:
                        tmp_res.append(word)
                    
            res.append(tmp_res)
        return res

    # Used for generate dependencies
    def genDependency(self, stringList):
        
        count = 0
        
        global parser
        
        dependRes = []
        
        for i in stringList:
            for j in i:
                try:
                    ggg = parser.parse(j).__next__()
                    for tup in list(ggg.triples()):
                        #if 'amod' in tup or 'advmod' in tup or 'xcomp' in tup:
                        if 'amod' in tup:
                            dependRes.append([tup[0][0], tup[2][0]])
                except:
                    print ("Still generating dependencies...")
                    break
            count += 1
            print (count)
        
        return dependRes

    def modelGenerator(self, sent):

        if len(sent) == 0:
            print ("Input is none...")
            return 

        tmp_list = []

        if isinstance(sent, list):
            for i in sent:
                tmp_list.append(self.cleanSent(sent))

        elif isinstance(sent, str):
            tmp_list.append(self.cleanSent(sent))

        else:
            print ("Input format error...")
            return

        print ("Generating dependencies...")
        tmp_dep_list = self.genDependency(tmp_list)
        print ("Generating dependencies ok")
        # print (tmp_dep_list)
        
        print ("Training model...")
        if os.path.isfile(self.model_name):
            print ("model exists, doing additional training...")
            self.model = word2vec.Word2Vec.load(self.model_name)
            self.model.build_vocab(tmp_dep_list, update=True)
        else:
            print ("model doesn't exists, doing initial training...")
            self.model.build_vocab(tmp_dep_list)
        self.model.train(tmp_dep_list, total_examples = self.model.corpus_count, epochs = 1000)
        self.model.save(self.model_name)
        print ("Training ok...")

        self.stringList += tmp_list

        for g in tmp_dep_list:
            if g[0] in list(self.model.wv.vocab.keys()) and g[1] in list(self.model.wv.vocab.keys()):
                self.depList.append([g[0], g[1]])

    def modelSave(self):

        self.model.save(self.model_name)
        print (self.model_name + " saves successfully!")

    def tagGenerator(self):

        words = [np.array(list(self.model[i[0]]) + list(self.model[i[1]])) for i in self.depList]
        vectors = StandardScaler().fit_transform(words)
        dbs = DBSCAN(eps=10, min_samples=5).fit(vectors) # parameter
        labels = dbs.labels_

        clusters = {}

        for lab, dep in zip(labels, self.depList):
            if lab not in clusters:
                clusters[lab] = [dep]
            else:
                clusters[lab].append(dep)

        # tag of a cluster depended on the highest frequency
        tag = []

        for i in clusters.values():
            maxSim = 0
            for val in i:
                tmp_maxSim = self.model.wv.vocab[val[0]].count + self.model.wv.vocab[val[1]].count
                if tmp_maxSim >= maxSim:
                    tmp_tag = " ".join([val[0], val[1]])
            tag.append(tmp_tag)

        return tag

if __name__ == '__main__':
    df = open("590_18_suggestion_IMBA.txt", "r") 
    input_list = []
    for i in df.readlines():
        input_list.append(i)
    print (input_list)

    # Below is how to use the function
    tag = tagGenerator("suggestion_IMBA") # "suggestion_IMBA" refers to the type of survey results
    tag.modelGenerator(input_list)
    tag.modelSave()
    print (tag.tagGenerator())



