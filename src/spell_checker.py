# -*- coding: utf-8 -*-
from nltk import ngrams
import inv_matrix_gen
import channel_prior
import edit_operations
import operator
from metaphone import doublemetaphone
from joblib import Parallel, delayed
import multiprocessing
import sys
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
    p = prior_prob[cand]
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
           #get_score(query,c,ph)
           #scores.append(get_score(query,c,ph))
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
bi_inv = inv_matrix_gen.load_obj("../data/bi_gram_inv")
tri_inv = inv_matrix_gen.load_obj("../data/tri_gram_inv")
prior_prob = inv_matrix_gen.load_obj("../data/prior_prob")
add = inv_matrix_gen.load_obj("../data/channel_data/add_X_Y")
dle = inv_matrix_gen.load_obj("../data/channel_data/del_X_Y")
rev = inv_matrix_gen.load_obj("../data/channel_data/rev_X_Y")
sub = inv_matrix_gen.load_obj("../data/channel_data/sub_X_Y")
qs=[]
with open(sys.argv[1],encoding = "ISO-8859-1") as ii:
    lines = [line.rstrip('\n') for line in ii]
    for i in lines:
        line=i.split("\t")
        qs.append(line[0])
print(qs)
#qs = ["belive", "bouyant", "comitte", "distarct","extacy", "failr", "hellpp", "gracefull", "liason", "ocassion", "possable", "thruout", "volly", "tatoos", "respe"]    
for query in qs:
    with open(sys.argv[2], 'a') as out:
        out.write(query+'\t')
        cand,scores=spell_checker(query)
        for i in range(len(cand)): 
            out.write(cand[i]+'\t'+str(scores[i])+'\t')
        out.write('\n')
   
