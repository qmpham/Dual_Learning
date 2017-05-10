import NMT
import LM
from NMT.dualnmt import dualnmt
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

dataset_bi_en = "/people/minhquang/Dual_NMT/data/train10/train10.en.tok"
dataset_bi_fr = "/people/minhquang/Dual_NMT/data/train10/train10.fr.tok"
dataset_mono_en = ""
dataset_mono_fr = ""
vocal_en = "/people/minhquang/Dual_NMT/data/train10/train10.en.tok.pkl"
vocal_fr = "/people/minhquang/Dual_NMT/data/train10/train10.fr.tok.pkl"
test_en = "/people/minhquang/Dual_NMT/data/test10/test10.en.tok"
test_fr = "/people/minhquang/Dual_NMT/data/test10/test10.fr.tok"
path = "/people/minhquang/Dual_NMT/models/model.iter30000.npz"


def dual_ascent(lr, tparams, grads, reward, optimizer_params = None):     
    
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_grad_shared' % k) \
                for k,p in tparams.iteritems() ]
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
    
    avg_reward = T.mean(reward)
    
    f_grad_shared = theano.function([reward], avg_reward, updates = g_up)
    
    params_up = [(p , p + lr * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    
    f_update = theano.function([lr], [], updates = params_up)
    
    return f_grad_shared, f_update


def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
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
        ):
    # Translation Model:
        
    # Model options
    model_options = OrderedDict(sorted(locals().copy().items()))


    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    
    model_options_fr_en = model_options.copy()
    model_options_en_fr = model_options.copy()
        
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
    new_cost = T.mean(reward * (-cost))
    
    tparams = nmt_en_fr.tparams
    inps = [x_bi, x_bi_mask, y_bi, y_bi_mask]
    # gradient newcost = gradient( reward * -cost) = sum reward_i * gradient( -cost_i) = reward_i * gradient(log p(s_mid | s))
    
    grad = T.grad(new_cost,wrt=theano_util.itemlist(tparams)) 

    f = theano.function(inps+[reward],grad)
    lr = T.scalar('lrate')
    f_grad_shared, f_update = dual_ascent(lr, tparams, grads, reward) 
    
    
    
    #nmt_fr_en.build_model()
    
    #build samplers
    #nmt_en_fr.build_sampler()
    
    
    #test samplers
    train_bi = NMT.data_iterator.TextIterator(dataset_bi_en, dataset_bi_fr,
                         [vocal_en], vocal_fr,
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         skip_empty=True,
                         shuffle_each_epoch = True,
                         sort_by_length=sort_by_length,
                         maxibatch_size=maxibatch_size)
    x_bi,y_bi = train_bi.next()
    x_bi, x_bi_mask, y_bi, y_bi_mask = prepare_data(x_bi, y_bi, maxlen=maxlen,
                                                    n_words_src=n_words_src,
                                                    n_words=n_words)
    #print(f(x_bi, x_bi_mask, y_bi, y_bi_mask, numpy.ones(x_bi_mask.shape[1],dtype=numpy.float32)))
    """
    for jj in xrange(numpy.minimum(5, x.shape[2])):
        stochastic = True
        x_current = x[:, :, jj][:, :, None]
        # remove padding
        x_current = x_current[:,:x_mask.astype('int64')[:, jj].sum(),:]
        
        #print(x_current)
        sample, score, sample_word_probs, alignment, hyp_graph = nmt_en_fr.gen_sample(
                                   x_current,
                                   k=5,
                                   maxlen=30,
                                   stochastic=stochastic,
                                   argmax=False,
                                   suppress_unk=False,
                                   return_hyp_graph=False)
        
        print 'Source ', jj, ': ',
        for pos in range(x.shape[1]):
            if x[0, pos, jj] == 0:
                break
            for factor in range(factors):
                vv = x[factor, pos, jj]
                if vv in nmt_en_fr.worddicts_r[factor]:
                    sys.stdout.write(nmt_en_fr.worddicts_r[factor][vv])
                else:
                    sys.stdout.write('UNK')
                if factor+1 < factors:
                    sys.stdout.write('|')
                else:
                    sys.stdout.write(' ')
        print
        print 'Truth ', jj, ' : ',
        for vv in y[:, jj]:
            if vv == 0:
                break
            if vv in nmt_en_fr.worddicts_r[-1]:
                print nmt_en_fr.worddicts_r[-1][vv],
            else:
                print 'UNK',
        print
        print 'Sample ', jj, ': ',
        for ss in sample:
            for vv in ss:
                if vv == 0:
                    break
                if vv in nmt_en_fr.worddicts_r[-1]:
                    print nmt_en_fr.worddicts_r[-1][vv],
                else:
                    print 'UNK',
            print "\n"
      
        for i in range(5):
            print(sample_word_probs[i])
            print(score[i])
            print(score[i] + numpy.log(sample_word_probs[i]).sum())"""
            
    
    return 0

train()
