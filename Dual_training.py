import NMT
import LM
from NMT.dualnmt import dualnmt
from LM.lm import lm
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import numpy
import copy
import argparse

import os
import warnings
import sys
import time

import itertools

from subprocess import Popen

from collections import OrderedDict
import NMT.data_iterator as data_iterator
import NMT.theano_util as theano_util
profile = False

dataset_bi_en = "/people/minhquang/Dual_NMT/data/train/train10/train10.en.tok"
dataset_bi_fr = "/people/minhquang/Dual_NMT/data/train/train10/train10.fr.tok"
dataset_mono_en = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok"
dataset_mono_fr = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok"
vocal_en = "/people/minhquang/Dual_NMT/data/train/train10/train10.en.tok.pkl"
vocal_fr = "/people/minhquang/Dual_NMT/data/train/train10/train10.fr.tok.pkl"
test_en = "/people/minhquang/Dual_NMT/data/validation/devel03/devel03.en.tok"
test_fr = "/people/minhquang/Dual_NMT/data/validation/devel03/devel03.fr.tok"
path_trans_en_fr = "/people/minhquang/Dual_NMT/models/NMT/model_hal.iter132500.npz"
path_trans_en_fr = ""
path_mono_en = ""
path_mono_fr = ""

def dual_ascent(lr, tparams, grads, inps, reward, optimizer_params = None):     
    
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_grad_shared' % k) \
                for k,p in tparams.iteritems() ]
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
    
    avg_reward = T.mean(reward)
    
    f_grad_shared = theano.function(inps + [reward], avg_reward, updates = g_up)
    
    params_up = [(p , p + lr * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    
    f_update = theano.function([lr], [], updates = params_up)
    
    return f_grad_shared, f_update

# batch preparation, returns padded batch and mask
def prepare_data_mono(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    # filter according to mexlen
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.

    return x, x_mask

def prepare_data_bi(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

def train(dim_word=512,  # word vector dimensionality
              dim=1000,  # the number of LSTM units
              factors=1, # input factors
              dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
              encoder='gru',
              decoder='gru_cond',
              patience=10,  # early stopping patience
              max_epochs=5000,
              finish_after=10000000,  # finish after this many updates
              dispFreq=1000,
              lrate=0.0001,  # learning rate
              n_words_src=None,  # source vocabulary size
              n_words=None,  # target vocabulary size
              maxlen=100,  # maximum length of the description
              optimizer='adam',
              batch_size=16,
              valid_batch_size=16,
              saveto='models/model_dual.npz',
              validFreq=10000,
              saveFreq=30000,   # save the parameters after every saveFreq updates
              sampleFreq=10000,   # generate some samples after every sampleFreq
              use_dropout=False,
              dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
              dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
              dropout_source=0, # dropout source words (0: no dropout)
              dropout_target=0, # dropout target words (0: no dropout)
              reload_=False,
              reload_training_progress=True, # reload trainig progress (only used if reload_ is True)
              overwrite=False,
              external_validation_script=None,
              sort_by_length=True,
              maxibatch_size=20, #How many minibatches to load at one time
              model_version=0.1, #store version used for training for compatibility
              prior_model=None, # Prior model file, used for MAP
              tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
              tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
              encoder_truncate_gradient=-1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
              decoder_truncate_gradient=-1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
              alpha = 0.5
        ):
    # Translation Model:
        
    # Model options
    
    model_options_trans = OrderedDict(sorted(locals().copy().items()))
    model_options_mono = OrderedDict()
    
    if model_options_trans['dim_per_factor'] == None:
        if factors == 1:
            model_options_trans['dim_per_factor'] = [model_options_trans['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(model_options_trans['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options_trans['dim_per_factor']) == model_options_trans['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    
    model_options_fr_en = model_options_trans.copy()
    model_options_en_fr = model_options_trans.copy()
        
    model_options_fr_en["datasets_bi"] = [dataset_bi_fr,dataset_bi_en]
    model_options_fr_en["datasets_mono"] = dataset_mono_fr
    model_options_fr_en["dictionaries"] = [vocal_fr,vocal_en]
    
    model_options_en_fr["datasets_mono"] = dataset_mono_en
    model_options_en_fr["datasets_bi"] = [dataset_bi_en,dataset_bi_fr]
    model_options_en_fr["dictionaries"] = [vocal_en,vocal_fr]
    
    # Intilize params and tparams
    nmt_en_fr = dualnmt()
    #nmt_fr_en = dualnmt()
    
    nmt_en_fr.get_options(model_options_en_fr)   
    #nmt_fr_en.get_options(model_options_fr_en)
    
    nmt_en_fr.invert_dict()
    #nmt_fr_en.invert_dict()
    
    nmt_en_fr.init_params()
    #nmt_fr_en.init_params()
    
    #load params
    
    nmt_en_fr.load_params(path)
    
    # build models
    trng, use_noise, x_bi, x_bi_mask, y_bi, y_bi_mask, opt_ret, cost = nmt_en_fr.build_model()
    
    # Compute gradient
    reward = T.vector("reward")
    
    # -cost = log(p(s_mid|s))
    new_cost = T.mean(reward * (-cost))
    
    tparams = nmt_en_fr.tparams
    inps = [x_bi, x_bi_mask, y_bi, y_bi_mask]
    # gradient newcost = gradient( reward * -cost) = avg reward_i * gradient( -cost_i) = avg reward_i * gradient(log p(s_mid | s)) stochastic approxiamtion of policy gradient
    
    grad = T.grad(new_cost,wrt=theano_util.itemlist(tparams)) 

    f = theano.function(inps+[reward],grad)
    #build f_grad_shared: average rexards, f_update: update params by gradient newcost
    lr = T.scalar('lrate')
    f_grad_shared_en_fr, f_update_en_fr = dual_ascent(lr, tparams, grad, inps, reward) 
    
        
    #build samplers
    nmt_en_fr.build_sampler()
    
    
    
    #build language model
    model_options_mono['encoder'] = 'gru'
    model_options_mono['dim'] = 1000
    model_options_mono['dim_word'] = 100
    model_options_mono['n_words'] = 100000
    lm_en = lm()
    lm_fr = lm()
    
    lm_en.get_options(model_options_mono)
    lm_fr.get_options(model_options_mono)
    
    lm_en.init_params()
    #lm_fr.init_params()
    
    lm_en.init_tparams()
    #lm_fr.init_tparams()
    
    lm_en.build_model()
    
    
    # load language model's parameters
    #lm_en.load_params("")
    print lm_en.params
    
    
    #Soft-landing phrase
    train_bi_en_fr = NMT.data_iterator.TextIterator(dataset_bi_en, dataset_bi_fr,
                             [vocal_en], vocal_fr,
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             skip_empty=True,
                             shuffle_each_epoch = True,
                             sort_by_length=sort_by_length,
                             maxibatch_size=maxibatch_size)
    train_bi_en_fr = NMT.data_iterator.TextIterator(dataset_bi_fr, dataset_bi_en,
                             [vocal_en], vocal_fr,
                             n_words_source=n_words_src, n_words_target=n_words,
                             batch_size=batch_size,
                             maxlen=maxlen,
                             skip_empty=True,
                             shuffle_each_epoch = True,
                             sort_by_length=sort_by_length,
                             maxibatch_size=maxibatch_size)
    train_mono_en = LM.data_iterator.TextIterator(dataset_mono_en,dict_mono_en)
    for i in range(max_epochs):
        for x_bi,y_bi in train_bi:
            x_bi, x_bi_mask, y_bi, y_bi_mask = prepare_data_bi(x_bi, y_bi, maxlen=maxlen,
                                                        n_words_src=n_words_src,
                                                        n_words=n_words)
        #print(f(x_bi, x_bi_mask, y_bi, y_bi_mask, numpy.ones(x_bi_mask.shape[1],dtype=numpy.float32)))
        
            for jj in xrange(numpy.minimum(5, x_bi.shape[2])):
                stochastic = True
                x_current = x_bi[:, :, jj][:, :, None]
                # remove padding
                x_current = x_current[:,:x_bi_mask.astype('int64')[:, jj].sum(),:]
            
                #print(x_current)
                sample, score, sample_word_probs, alignment, hyp_graph = nmt_en_fr.gen_sample(
                                       x_current,
                                       k=5,
                                       maxlen=30,
                                       stochastic=stochastic,
                                       argmax=False,
                                       suppress_unk=False,
                                       return_hyp_graph=False)
        
        
        
        
        
       
            
    
    return 0

train()
