# -*- coding: utf-8 -*-
from nltk import ngrams
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def inv_matrix_gen(n, path):
    word_map = dict()
    for word in prior_freq:
        n_grams = ngrams(word,n)
        for i in n_grams:
            if not (i in word_map):
                word_map[i] = [word]
            else:
                words_mapped = word_map.get(i)
                words_mapped.append(word)
                word_map[i] = words_mapped
    
    n_gram_inv_matrix = dict()
    for n_gram in word_map:
        words = word_map[n_gram]
        n_gram_one_map = dict()
        len_set = set([len(word) for word in words])
        for i in len_set:
            word_len_set= [word for word in words if len(word)==i]
            n_gram_one_map[i] = word_len_set 
        n_gram_inv_matrix[n_gram] = n_gram_one_map
            
    save_obj(n_gram_inv_matrix,path)


#==============================================================================
# Reading the whole dcitionary for preprocessing
#==============================================================================
with open("../data/BaseDictionary.txt") as file:
    lines = [line.rstrip('\t\n') for line in file]
#==============================================================================
# Parsing the frequencies
#==============================================================================
prior_freq = dict()
for line in lines:
    sentence = line.split('\t')
    word = sentence[0]
    freq = int(sentence[1])
#    Add 1 smoothing
    prior_freq[word] = freq + 1

total_freq = sum(prior_freq.values())
#==============================================================================
# Calculating and saving prior probabilities
#==============================================================================
prior_prob = dict()
for word in prior_freq:   
    prior_prob[word] = prior_freq[word] / total_freq

save_obj(prior_prob,"../data/prior_prob")

#==============================================================================
# Inverted Bigram matrix
#==============================================================================

inv_matrix_gen(2 ,"../data/bi_gram_inv")
inv_matrix_gen(3 ,"../data/tri_gram_inv")