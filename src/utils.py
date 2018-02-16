# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import inspect
import argparse
import subprocess
import numpy as np
from logging import getLogger

#from .logger import create_logger
#from dictionary import Dictionary
from collections import OrderedDict

MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')


logger = getLogger()


# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False

def get_nn_avg_dist(src, tgt, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    #print(FAISS_AVAILABLE)
    #if FAISS_AVAILABLE:        
    if hasattr(faiss, 'StandardGpuResources'):
        # gpu mode
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        index = faiss.GpuIndexFlatIP(res, tgt.shape[1], config)
        logger.info("faiss gpu mode!")
    else:
        # cpu mode
        index = faiss.IndexFlatIP(tgt.shape[1])
    index.add(src)
    distances, _ = index.search(src, knn)
    return distances.mean(1)
    """
    else:
        bs = 1024
        all_distances = []
        tgt = tgt.transpose()
        for i in range(0, src.shape[0], bs):
            distances = src[i:i + bs].dot(tgt)
            ind = distances.argsort()[:,-knn:].tolist()
            for j in range(distances.shape[0]):                            
                all_distances.append(distances[j,:][ind[j]].mean())
        all_distances = np.array(all_distances)
        return all_distances
    """

def id2id_trans_emb(dict_transmod, dict_emb): # OrderedDict() is demanded
    assert type(dict_transmod) == OrderedDict
    assert type(dict_emb) == OrderedDict
    ind_common = []
    id2id = []
    c = 0
    n = 0
    for w in dict_transmod:
        n = n + 1
        if w in dict_emb:
            ind_common.append(1.0)
            id2id.append(dict_emb[w])
            c = c + 1
        else:
            ind_common.append(0)
            id2id.append(0)
    id2id = np.array(id2id).astype("int64")
    ind_common = np.array(ind_common).astype("float32")
    print("covering percentage(transmod_emb): %f"%(float(c)/float(n)))
    return id2id, ind_common
    
def id2id_emb_idf(dict_emb, dict_idf): # OrderedDict() is demanded
    assert type(dict_idf) == OrderedDict
    assert type(dict_emb) == OrderedDict
    ind_common = []
    id2id = []
    c = 0
    n = 0
    for w in dict_emb:
        n = n + 1
        if w in dict_idf:
            ind_common.append(np.float32(1.0))
            id2id.append(dict_idf[w])
            c = c + 1
        else:
            ind_common.append(np.float32(0.0))
            id2id.append(0)
    id2id = np.array(id2id).astype("int64")
    ind_common = np.array(ind_common).astype("float32")
    print("covering percentage(emb_idf): %f"%(float(c)/float(len(dict_idf.keys()))))
    return id2id, ind_common

def bow_idf(sentences, ind_common, id2id, idf, emb):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        tf_idf = []
        embeds = []
        """
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        """
        for id in sent:
            emb_id = id2id[0][id]
            idf_ind = ind_common[0][id] * ind_common[1][emb_id]
            idf_id = id2id[1][emb_id]
            if idf_ind !=0:
                tf_idf.append(idf[idf_id])
                embeds.append(emb[emb_id,:] * idf[idf_id])
        if len(tf_idf)==0:
            tf_idf.append(1.0)
            embeds.append(emb[0,:])        
        embeddings.append(np.sum(embeds, axis=0)/np.sum(tf_idf))
    return np.vstack(embeddings)

def tf_idf(sentences, ind_common, id2id, idf, emb_dict_size):
    """
    Get idf index for bag of words:
    """
    tf_idf = np.zeros((len(sentences),emb_dict_size))
    c = 0
    for sent in sentences:        
        for id in sent:
            emb_id = id2id[0][id]
            idf_ind = ind_common[0][id] * ind_common[1][emb_id]
            idf_id = id2id[1][emb_id]
            tf_idf[c,emb_id] =  idf[idf_id] * idf_ind
        if tf_idf[c,:].sum() == 0:
           tf_idf[c,0] = 1.0
        # averaging tf_idf
        c = c + 1    
    tf_idf = tf_idf/tf_idf.sum(1)[:,None]
    return tf_idf.astype("float32")

def get_idf(path, src_lg, tgt_lg, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_lg: OrderedDict(), tgt_lg: OrderedDict()}
    idf_ = {src_lg: [], tgt_lg: []}
    word2id = {src_lg: OrderedDict(), tgt_lg: OrderedDict()}
    k = 0
    data = load_data(path, src_lg, tgt_lg)
      
    for lg in idf:
        #start_idx = 200000 + k * n_idf
        #end_idx = 200000 + (k + 1) * n_idf
        for sent in data[lg]:#[start_idx:end_idx]:
            for word in set(sent):
                idf[lg][word] = idf[lg].get(word, 0) + 1
        n_doc = len(data[lg])#[start_idx:end_idx])
        id = 0        
        for word in idf[lg]:
            word2id[lg][word] = id
            idf_[lg].append(max(1, np.log10(n_doc / (idf[lg][word]))))
            id = id + 1
        k += 1
             
    return idf_, word2id
    
def read_embeddings(path, dim=None, n_max=1e9):
    """
    Read all words from a word embedding file, and optionally filter them.
    """
    word2id = OrderedDict()
    embeddings = []
    with open(path, 'r') as f:
        line = f.readline()
        dim = int(line.split(' ', 1)[1])
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                logger.warning('Word "%s" has several embeddings!' % word)
                continue
            word2id[word] = len(word2id)
            embeddings.append(np.fromstring(vec, sep=' '))
            if len(word2id) == n_max:
                break
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]
    logger.info("Found %s word vectors of size %s" % (len(word2id), dim))
    return word2id, embeddings
    
def load_data(path, lg1, lg2, n_max=1e10, lower=False):
    """
    Load data parallel sentences
    """   
    # load sentences
    data = {lg1: [], lg2: []}
    for lg in [lg1, lg2]:
        fname = path[lg]
        with open(fname) as f:
            for i, line in enumerate(f):
                if i >= n_max:
                    break
                line = line.lower() if lower else line
                data[lg].append(line.rstrip().split())
    
    # get only unique sentences for each language
    assert len(data[lg1]) == len(data[lg2])
    data[lg1] = np.array(data[lg1])
    data[lg2] = np.array(data[lg2])

    logger.info("Loaded data %s-%s (%i sentences)." % (lg1, lg2, len(data[lg1])))
    return data
    
