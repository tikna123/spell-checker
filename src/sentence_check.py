from __future__ import division
import nltk
from nltk.util import ngrams
##dl = nltk.downloader.Downloader("http://nltk.github.com/nltk_data/")
##dl.download() choose corpora and then installl brown
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import re
import string
from collections import Counter
from collections import OrderedDict
import operator
import pickle
import sys
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
###spell checker

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# -*- coding: utf-8 -*-
from nltk import ngrams
import inv_matrix_gen
import channel_prior
import edit_operations
import operator
from metaphone import doublemetaphone
from joblib import Parallel, delayed
import multiprocessing

def get_ngram_candidates(query,n,inv_matrix):
    
    candidates = []
    n_grams = ngrams(query,n)
        
    for i in n_grams:
        if i in inv_matrix:
            for lens in inv_matrix[i]:
                if lens > len(query) - 3 and lens < len(query) + 4:
#                    print(inv_matrix[i][lens])
                    candidates.extend(inv_matrix[i][lens])
    return candidates

def get_candidates(query):
    query= query.upper()
#==============================================================================
#     Get all trigram candidates
#==============================================================================
    tri_candidates = get_ngram_candidates(query,3,tri_inv)
    bi_candidates = get_ngram_candidates(query,2,bi_inv)
#    print(tri_candidates)
    tri_candidates.extend(bi_candidates)
#    print(len(bi_candidates))
    return tri_candidates
    

def channel_operation_prob(operation):
    if operation == "":
        return 1
    split_opr = operation.split(":")
    op_code = split_opr[0]
    operands = split_opr[1]
    if op_code == "S" and operands[0] == operands[1]:
        return 1
    i = ord(operands[0]) - ord("A")
    j = ord(operands[1]) - ord("A") - 1
      
#    print(i)
#    print(j)
    if op_code == "D":
        if j >=26:
            return 1
        return dle[i][j]
    if op_code == "I":
        if j >=26:
            return 1
        return add[i][j]
    if op_code == "S":
        return sub[i][j]
    if op_code == "T":
        return rev[i][j]
        
def get_channel_prob(query, candidate):

    query = query.upper()
    candidate = candidate.upper()
    
    edit_string = edit_operations.edit_opr(query,candidate)
    edit_list = edit_string.split("_")
    
    channel_prob = 1
    for op in edit_list:
        channel_prob = channel_prob * channel_operation_prob(op)        

    return channel_prob

def phonetic_score(query,cand):
    phn_query = doublemetaphone(query)
    phn_cand = doublemetaphone(cand)
    
    score = 1
    if phn_query[0] == phn_cand[0]:
        return 1000000
    elif phn_query[1] == phn_cand[0] or phn_query[0] == phn_cand[1]:
        return 1000
    elif phn_query[1] == phn_cand[1]:
        return 10
    else:
        return 1
    return score    
    
def get_score(query,cand,ph=True):
    p = prior_prob[cand.lower()]
    c = get_channel_prob(query,cand)
    s = 1    
    if ph:    
        s = phonetic_score(query,cand)
    return p*c*s

def spell_checker(query,ph=True):
    if len(query) < 2:
        results = [query]
        resultscores = [1]
    else:
        candis = get_candidates(query)
    # To make parallel
    #    print(len(candis))
    #    print(len(set(candis)))

        candis = set(candis)
        #print(candis)   
        scores = Parallel(n_jobs=num_cores)(delayed(get_score)(query,c,ph) for c in candis)
        #scores=[]
        #for c in candis:
         #  get_score(query,c,ph)
          # scores.append(get_score(query,c,ph))
        scores, candis = zip(*sorted(zip(scores, candis),reverse=True))
        #print(scores)
        results = candis[:10]
        resultscores = scores[:10]
        #print("%s\t" % (query), end="")
        #print("%s" % (results[0]), end="")
        #for x in results[1:]:
        #print(",%s" % (x), end="")
        #print()
    return results,resultscores
    

num_cores = multiprocessing.cpu_count()
bi_inv = load_obj("../data/bi_gram_inv")
tri_inv = load_obj("../data/tri_gram_inv")
#prior_prob = inv_matrix_gen.load_obj("../data/prior_prob")
add = load_obj("../data/channel_data/add_X_Y")
dle = load_obj("../data/channel_data/del_X_Y")
rev = load_obj("../data/channel_data/rev_X_Y")
sub = load_obj("../data/channel_data/sub_X_Y")

#query = "jallo"
#qs = ["belive", "bouyant", "comitte", "distarct","extacy", "failr", "hellpp", "gracefull", "liason", "ocassion", "possable", "thruout", "volly", "tatoos", "respe"]    
#for query in qs:

#spell_checker(query)
  
   


##COCA dataset
##Open datasets and get 2,3,4 and 5 grams
w2gram={}
w3gram={}
w4gram={}
w5gram={}

##open datasets of COCA
with open("COCA/w2_.txt",encoding = "ISO-8859-1") as ff:
    lines = [line.rstrip('\n') for line in ff]
    for i in lines:
        line=i.split("\t")
        w2gram[line[1],line[2]]=float(line[0])
with open("COCA/w3_.txt",encoding = "ISO-8859-1") as gg:
    lines = [line.rstrip('\n') for line in gg]
    for i in lines:
        line=i.split("\t")
        w3gram[line[1],line[2],line[3]]=float(line[0])
with open("COCA/w4_.txt",encoding = "ISO-8859-1") as hh:
    lines = [line.rstrip('\n') for line in hh]
    for i in lines:
        line=i.split("\t")
        w4gram[line[1],line[2],line[3],line[4]]=float(line[0])
with open("COCA/w5_.txt",encoding = "ISO-8859-1") as ii:
    lines = [line.rstrip('\n') for line in ii]
    for i in lines:
        line=i.split("\t")
        w5gram[line[1],line[2],line[3],line[4],line[5]]=float(line[0])

##Sentence.tsv #works both for phrases and sentences
##Tokenize sentences
trainsentences=[]
with open(sys.argv[1]) as f:
    lines = [line.rstrip('\n') for line in f]
    for i in lines:
        g=i.split(". ") #change the separator based on the delimiter between worng and right sentences
        trainsentences.append(word_tokenize(g[0]))
##Eliminate punctuations #if you pass sentence list with punctuations it will return sentence list without punctuations
def eliminatepunctuation(sentencelist):
    x=re.compile('[%s]' % re.escape(string.punctuation))
    withoutpunctuation=[]
    for sent in sentencelist:
        newsent=[]
        for word in sent:
            newword=x.sub(r'u''',word)
            if not newword == u'':
                newsent.append(newword.lower())
        withoutpunctuation.append(newsent)
    return withoutpunctuation


##Extract n-grams #if you pass list of words in a sentence it will return the different ngrams 2,3,4 and 5
def extractngrams(listofwords):    
    gram_1=ngrams(listofwords,2)
    gram_2=ngrams(listofwords,3)
    gram_3=ngrams(listofwords,4)
    gram_4=ngrams(listofwords,5)
    return gram_1,gram_2,gram_3,gram_4
trainsentenceswithoutpnctuation=eliminatepunctuation(trainsentences)
###if word is not there in dictionary error word:
#==============================================================================
# Reading the whole dcitionary for preprocessing
#==============================================================================
with open("BaseDictionary.txt") as file:
    lines = [line.rstrip('\t\n') for line in file]
#==============================================================================
# Parsing the frequencies
#==============================================================================
prior_freq = dict()
for line in lines:
    sentence = line.split('\t')
    word = sentence[0].lower()
    freq = int(sentence[1])
#    Add 1 smoothing
    prior_freq[word.lower()] = freq + 1

total_freq = sum(prior_freq.values())
#==============================================================================
# Calculating and saving prior probabilities
#==============================================================================
prior_prob = dict()
for word in prior_freq:   
    prior_prob[word.lower()] = prior_freq[word.lower()] / total_freq
save_obj(prior_prob,"../data/prior_prob")
for sentence in trainsentenceswithoutpnctuation:
    errorbasedondict=[]
    ngramcount={}
    for words in sentence:
        if words not in prior_freq.keys():
            errorbasedondict.append(words)
        ngramcount[words]=0
    #print(errorbasedondict)
    g1,g2,g3,g4=extractngrams(sentence)
####find error words based on ngrams
    errorbasedonngram=[]
    errorundeductable=[]
    ##Compute MLE
    for j in g1:
        if j in w2gram.keys():
            for words in j:
                ngramcount[words]+=w2gram[j]
    for j in g2:
        if j in w3gram.keys():
            for words in j:
                ngramcount[words]+=w3gram[j]
    for j in g3:
        if j in w4gram.keys():
            for words in j:
                ngramcount[words]+=w4gram[j]
    for j in g4:
        if j in w5gram.keys():
            for words in j:
                ngramcount[words]+=w5gram[j]
    cc=Counter(ngramcount)
    sorted_x = sorted(cc.items(), key=operator.itemgetter(1))
    #print(sorted_x)
    value1 = [k for (k,v) in sorted_x if v == 0]
    #print(value1)
    #print(set(errorbasedonngram))
    #print(set(errorbasedonngram))
    ##Finding candidate words which are erroneous in the sentence and then finding appropriate words using spell check
    if len(errorbasedondict) != 0:
        Candidatewordstocheckfor=errorbasedondict
    else:
        Candidatewordstocheckfor=list(set(value1))
    #print(Candidatewordstocheckfor)
    if len(Candidatewordstocheckfor) == 0:
        cor,scores=spell_checker(sorted_x[0][0])
        prioritydic={}
        ###Maximizing Likelihood function
        for i in cor:
            scor=scores[cor.index(i)]
            sentence1 = sentence[:]
            word=i.lower()
            sentence1[sentence1.index(sorted_x[0][0])]=word
            g1,g2,g3,g4=extractngrams(sentence1)
            ##Compute MLE
            Overallsum=[]
            for j in g1:
                #print(j)
                #j1=(b.replace(word) for i in j)
                #print j1
                if j in w2gram.keys():
                    value = [v for (k,v) in w2gram.items() if j[0] in k[0]]
                    if sum(value) != 0:
                        Overallsum.append((w2gram[j]/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
                else:
                    value = [v for (k,v) in w2gram.items() if j[0] in k[0]]
                    if sum(value) != 0:
                        Overallsum.append((1/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
            for j in g2:
                if j in w3gram.keys():
                    value = [v for (k,v) in w3gram.items() if j[0] in k[0]]
                    #print(len(value))
                    if sum(value) != 0:
                        Overallsum.append((w3gram[j]/sum(value))*scor)
                    else:
                        #print(j[0])
                        errorbasedonngram.append(j[0])
                else:
                    value = [v for (k,v) in w3gram.items() if j[0] in k[0]]
                    if sum(value) != 0:
                        Overallsum.append((1/sum(value))*scor)
                    else:
                        #print(j[0])
                        errorbasedonngram.append(j[0])
            for j in g3:
                if j in w4gram.keys():
                    value = [v for (k,v) in w4gram.items() if j[0] in k[0]]
                    #print(len(value))
                    if sum(value) != 0:
                        Overallsum.append((w4gram[j]/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
                else:
                    value = [v for (k,v) in w4gram.items() if j[0] in k[0]]
                    if sum(value) != 0:
                        Overallsum.append((1/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
            for j in g4:
                if j in w5gram.keys():
                    value = [v for (k,v) in w5gram.items() if j[0] in k[0]]
                    #print(len(value))
                    if sum(value) != 0:
                        Overallsum.append((w5gram[j]/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
                else:
                    value = [v for (k,v) in w5gram.items() if j[0] in k[0]]
                    if sum(value) != 0:
                        Overallsum.append((1/sum(value))*scor)
                    else:
                        errorbasedonngram.append(j[0])
            prioritydic[word]=sum(Overallsum)
        #print(prioritydic)
        sortedpriority=Counter(prioritydic)
        sorted_priority = sorted(sortedpriority.items(), key=operator.itemgetter(1))
        new_sorted=sorted_priority[::-1]
        with open(sys.argv[1], 'a') as out:
            out.write(i+'\t')
            for sort in new_sorted[:3]:
                out.write(sort[0]+'\t'+str(sort[1])+'\t')
            out.write('\n')
    else:
        for i in Candidatewordstocheckfor:
            b,sc=spell_checker(i)
            prioritydic={}
            ###Maximizing Likelihood function
            for c in b:
                s=sc[b.index(c)]
                sentence1 = sentence[:]
                #print(c)
                word=c.lower()
                sentence1[sentence1.index(i)]=word
                g1,g2,g3,g4=extractngrams(sentence1)
                ##Compute MLE
                Overallsum=[]
                for j in g1:
                    #print(j)
                    #j1=(b.replace(word) for i in j)
                    #print j1
                    if j in w2gram.keys():
                        value = [v for (k,v) in w2gram.items() if j[0] in k[0]]
                        if sum(value) != 0:
                            Overallsum.append((w2gram[j]/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                    else:
                        value = [v for (k,v) in w2gram.items() if j[0] in k[0]]
                        if sum(value) != 0:
                            Overallsum.append((1/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                for j in g2:
                    if j in w3gram.keys():
                        value = [v for (k,v) in w3gram.items() if j[0] in k[0]]
                        #print(len(value))
                        if sum(value) != 0:
                            Overallsum.append((w3gram[j]/sum(value))*s)
                        else:
                            #print(j[0])
                            pass#Overallsum.append(s)
                    else:
                        value = [v for (k,v) in w3gram.items() if j[0] in k[0]]
                        if sum(value) != 0:
                            Overallsum.append((1/sum(value))*s)
                        else:
                            #print(j[0])
                            pass#Overallsum.append(s)
                for j in g3:
                    if j in w4gram.keys():
                        value = [v for (k,v) in w4gram.items() if j[0] in k[0]]
                        #print(len(value))
                        if sum(value) != 0:
                            Overallsum.append((w4gram[j]/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                    else:
                        value = [v for (k,v) in w4gram.items() if j[0] in k[0]]
                        if sum(value) != 0:
                            Overallsum.append((1/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                for j in g4:
                    if j in w5gram.keys():
                        value = [v for (k,v) in w5gram.items() if j[0] in k[0]]
                        #print(len(value))
                        if sum(value) != 0:
                            Overallsum.append((w5gram[j]/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                    else:
                        value = [v for (k,v) in w5gram.items() if j[0] in k[0]]
                        if sum(value) != 0:
                            Overallsum.append((1/sum(value))*s)
                        else:
                            pass#Overallsum.append(s)
                prioritydic[word]=sum(Overallsum)
            sortedpriority=Counter(prioritydic)
            sorted_priority = sorted(sortedpriority.items(), key=operator.itemgetter(1))
            new_sorted=sorted_priority[::-1]
            with open(sys.argv[2], 'a') as out:
                out.write(i+'\t')
                for sort in new_sorted[:3]:
                    out.write(sort[0]+'\t'+str(sort[1])+'\t')
                out.write('\n')
        




