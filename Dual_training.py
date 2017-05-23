import NMT
import LM
from NMT.dualnmt import dualnmt
from LM.lm import lm
import theano
import NMT.optimizers as optimizers
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.nanguardmode import NanGuardMode
import cPickle as pkl
import json
import numpy
import copy
import argparse
import ipdb 
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

dataset_bi_en = "/people/minhquang/nematus-master/data/train/train10/train10.en.tok.shuf"
dataset_bi_fr = "/people/minhquang/nematus-master/data/train/train10/train10.fr.tok.shuf"
dataset_mono_en = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok"
dataset_mono_fr = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok"
vocal_en = "/people/minhquang/Dual_NMT/data/train/train10/concatenated.en.tok.pkl"
vocal_fr = "/people/minhquang/Dual_NMT/data/train/train10/concatenated.fr.tok.pkl"
test_en = "/people/minhquang/Dual_NMT/data/validation/hit/hit.en.tok.shuf.dev.tok"
test_fr = "/people/minhquang/Dual_NMT/data/validation/hit/hit.fr.tok.shuf.dev.tok"
path_trans_en_fr = "/people/minhquang/Dual_NMT/models/NMT/model_en_fr.npz.dev.npz"
path_trans_fr_en = "/people/minhquang/Dual_NMT/models/NMT/model_fr_en.npz.dev.npz"
path_mono_en = "/people/minhquang/Dual_NMT/models/LM/model_lm_en.npz"
path_mono_fr = "/people/minhquang/Dual_NMT/models/LM/model_lm_fr.npz"


def dual_ascent(lr, tparams, grads, inps, reward, avg_reward, direction):     
    
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
        
    f_grad_shared = theano.function(inps + [reward], avg_reward, updates = g_up, on_unused_input='ignore')
    
    params_up = [(p , p + lr * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    
    f_update = theano.function([lr], [], updates = params_up, on_unused_input='ignore')
    
    return f_grad_shared, f_update

def dual_second_ascent(lr, alpha, tparams, grads, inps, reward, avg_reward, direction):
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_second_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    g_up = [(g1,g2) for g1,g2 in zip(g_shared,grads)]
    f_grad_shared = theano.function(inps+[reward],avg_reward,updates = g_up)
    params_up = [(p,p + lr * (1-alpha) * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    f_update = theano.function([lr],[],updates = params_up)
    return f_grad_shared, f_update
# batch preparation, returns padded batch and mask
def prepare_data_mono(seqs_x, maxlen=None, n_words=14000):
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

def prepare_data_bi(seqs_x, seqs_y, maxlen=None, n_words_src=14000,
                 n_words=14000):
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



N = 14000

def train(dim_word=512,  # word vector dimensionality
              dim=1000,  # the number of LSTM units
              factors=1, # input factors
              dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
              encoder='gru',
              decoder='gru_cond',
              lrate=0.0001,  # learning rate
              lrate_bi = 0.0001,
              n_words_src= N,  # source vocabulary size
              n_words= N ,  # target vocabulary size
              maxlen=50,  # maximum length of the description
              dispFreq = 1,
              batch_size=30,
              valid_batch_size=60,
              saveto='models/model_dual.npz',
              use_dropout=False,
              use_second_update = False,
              dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
              dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
              dropout_source=0, # dropout source words (0: no dropout)
              dropout_target=0, # dropout target words (0: no dropout)
              reload_=False,
              tie_encoder_decoder_embeddings=False, # Tie the input embeddings of the encoder and the decoder (first factor only)
              tie_decoder_embeddings=False, # Tie the input embeddings of the decoder with the softmax output embeddings
              encoder_truncate_gradient=-1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
              decoder_truncate_gradient=-1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
              alpha = 0.005,
              optional_update = True
        ):
    #hyperparameters:
    alp = theano.shared(numpy.float32(alpha),name="alpha")
    # Translation Model:
        
    # Model options
    u = time.time()
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
    model_options_fr_en["dictionaries"] = [vocal_fr,vocal_en]
    
    model_options_en_fr["datasets_bi"] = [dataset_bi_en,dataset_bi_fr]
    model_options_en_fr["dictionaries"] = [vocal_en,vocal_fr]
    
    # Intilize params and tparams
    nmt_en_fr = dualnmt()
    nmt_fr_en = dualnmt()
    
    nmt_en_fr.get_options(model_options_en_fr)   
    nmt_fr_en.get_options(model_options_fr_en)
    
    nmt_en_fr.invert_dict()
    nmt_fr_en.invert_dict()
    
    nmt_en_fr.init_params()
    nmt_fr_en.init_params()
    
    #load params
    
    nmt_en_fr.load_params(path_trans_en_fr)
    #nmt_fr_en.load_params(path_trans_fr_en)
    print "build nmt models ... ",
    # build models
    x_en, x_en_mask, y_fr, y_fr_mask, opt_ret_en_fr, cost_en_fr = nmt_en_fr.build_model()
    x_fr, x_fr_mask, y_en, y_en_mask, opt_ret_fr_en, cost_fr_en = nmt_fr_en.build_model()
    print "Done \n"
    # Compute gradient
    print "Build gradient ...",
    reward_en_fr = T.vector("reward_en_fr")
    reward_fr_en = T.vector("reward_fr_en")
    
    avg_reward_en_fr = T.mean(reward_en_fr)
    avg_reward_fr_en = T.mean(reward_fr_en)

    # -cost = log(p(s_mid|s))
    new_cost_en_fr = T.mean(reward_en_fr * (- cost_en_fr))
    new_cost_fr_en = T.mean(reward_fr_en * (- cost_fr_en))
    
    cost_ce_en_fr = cost_en_fr.mean()
    cost_ce_fr_en = cost_fr_en.mean()
        
    tparams_en_fr = nmt_en_fr.tparams
    tparams_fr_en = nmt_fr_en.tparams
    inps_en_fr = [x_en, x_en_mask, y_fr, y_fr_mask]
    inps_fr_en = [x_fr, x_fr_mask, y_en, y_en_mask]
    
    # gradient newcost = gradient( reward * -cost) = avg reward_i * gradient( -cost_i) = avg reward_i * gradient(log p(s_mid | s)) stochastic approximation of policy gradient
    
    grad_en_fr = T.grad(new_cost_en_fr, wrt=theano_util.itemlist(tparams_en_fr)) 
    grad_fr_en = T.grad(new_cost_fr_en, wrt=theano_util.itemlist(tparams_fr_en)) 
    
    grad_ce_en_fr = T.grad(cost_ce_en_fr, wrt=theano_util.itemlist(tparams_en_fr))
    grad_ce_fr_en = T.grad(cost_ce_fr_en, wrt=theano_util.itemlist(tparams_fr_en))
    
    g_en_fr = theano.function(inps_en_fr + [reward_en_fr], grad_en_fr)
    g_fr_en = theano.function(inps_fr_en + [reward_fr_en], grad_fr_en)
    
    #build f_grad_shared: average rewards, f_update: update params by gradient newcost
    lr_en_fr = T.scalar('lrate_en_fr')
    lr_fr_en = T.scalar('lrate_fr_en')
    lr1 = T.scalar('lrate1')
    lr2 = T.scalar('lrate2')
    
    f_dual_grad_shared_en_fr, f_dual_update_en_fr = dual_ascent(lr_en_fr, tparams_en_fr, grad_en_fr, inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr" ) 
    f_dual_grad_shared_fr_en, f_dual_update_fr_en = dual_ascent(lr_fr_en, tparams_fr_en, grad_fr_en, inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
    
    f_dual_second_grad_shared_en_fr, f_dual_second_update_en_fr = dual_second_ascent(lr_en_fr, alp, tparams_en_fr, grad_ce_en_fr, inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr")
    f_dual_second_grad_shared_fr_en, f_dual_second_update_fr_en = dual_second_ascent(lr_fr_en, alp, tparams_fr_en, grad_ce_fr_en, inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
        
    f_grad_shared_en_fr, f_update_en_fr, _ = optimizers.adadelta(lr1, tparams_en_fr, grad_ce_en_fr, inps_en_fr, cost_ce_en_fr)
    f_grad_shared_fr_en, f_update_fr_en, _ = optimizers.adadelta(lr2, tparams_fr_en, grad_ce_fr_en, inps_fr_en, cost_ce_fr_en)
    
    print "Done\n"
    #build samplers
    print "Build samplers ..."
    nmt_en_fr.build_sampler()
    nmt_fr_en.build_sampler()
    print "Done\n"
    
    #build language model
    model_options_mono['encoder'] = 'gru'
    model_options_mono['dim'] = 1024
    model_options_mono['dim_word'] = 512
    model_options_mono['n_words'] = N
    print "Build language models ...",
    lm_en = lm()
    lm_fr = lm()
    
    lm_en.get_options(model_options_mono)
    lm_fr.get_options(model_options_mono)
    
    lm_en.init_params()
    lm_fr.init_params()
    
    lm_en.init_tparams()
    lm_fr.init_tparams()
    
    lm_en.build_model()    
    lm_fr.build_model()
    
    # load language model's parameters
    lm_en.load_params(path_mono_en)
    lm_fr.load_params(path_mono_fr)
    print "Done"
    print "Compilation time:", time.time()-u
    #print lm_en.params
    
    
    #Soft-landing phrase   
    
    uidx = 0
    max_epochs = 500
    ud_start = time.time()
    for i in range(max_epochs):
        train_en = LM.data_iterator.TextIterator(dataset_mono_en, vocal_en, batch_size = batch_size /2,\
                                                 maxlen = 50, n_words_source = n_words_src)
        train_fr = LM.data_iterator.TextIterator(dataset_mono_fr, vocal_fr, batch_size = batch_size /2,\
                                                 maxlen = 50, n_words_source = n_words_src)
        train_en_fr = NMT.data_iterator.TextIterator(dataset_bi_en, dataset_bi_fr,\
                     [vocal_en], vocal_fr,\
                     batch_size=batch_size /2,\
                     maxlen=50,\
                     n_words_source=14000,\
                     n_words_target=14000)
        train_fr_en = NMT.data_iterator.TextIterator(dataset_bi_fr, dataset_bi_en,\
                     [vocal_fr], vocal_en,\
                     batch_size=batch_size /2,\
                     maxlen=50,\
                     n_words_source=14000,\
                     n_words_target=14000)
                 
        x_en = train_en.next()
        x_en_s, x_mask_en = prepare_data_mono(x_en, maxlen=maxlen,
                                                            n_words=n_words)
        x_fr = train_fr.next()
        x_fr_s, x_mask_fr = prepare_data_mono(x_fr, maxlen=maxlen,
                                                            n_words=n_words)
        
        x_en_en_fr, x_fr_en_fr = train_en_fr.next()
        x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = prepare_data_bi(x_en_en_fr, x_fr_en_fr,maxlen=maxlen,
                                                            n_words=n_words)
        x_fr_fr_en, x_en_fr_en = train_fr_en.next()
        x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = prepare_data_bi(x_fr_fr_en, x_en_fr_en,maxlen=maxlen,
                                                            n_words=n_words)
        while x_en_s is not None or x_fr_s is not None:
            uidx += 1 
        #Dual update
            # play game en->fr:
            
            if x_en_s is not None:
                s_source_en = []
                s_mid_fr = []
                s_mid_fr_2 = []
                reward = []
                u1 = time.time()
                for jj in xrange(x_en_s.shape[1]):
                    stochastic = True
                    x_current = x_en_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_en.astype('int64')[:, jj].sum(),:]
                    #sampling
                    sample, score, sample_word_probs, alignment, hyp_graph = nmt_en_fr.gen_sample(
                                           x_current,
                                           k=2,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=False,
                                           suppress_unk=True,
                                           return_hyp_graph=False)
                    tmp = []
                    for xs in x_en[jj]:
                        tmp.append([xs])
                    for ss in sample:
                        s_mid_fr.append(ss)
                        s_mid_fr_2.append(ss)
                        s_source_en.append(tmp)
                #print "time sampling one batch:", time.time() - u1
                u1 = time.time()
                s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask = prepare_data_bi(s_source_en, s_mid_fr)
                s_mid_fr_2, s_mid_fr_2_mask = prepare_data_mono(s_mid_fr_2)
                #print "time for prepare data: ", time.time() - u1
                #Time for dual ascent update: average over batch then over samples
                u1 = time.time()
                reward_en_fr = lm_fr.f_log_probs(s_mid_fr_2, s_mid_fr_2_mask) * alpha\
                                                + nmt_fr_en.f_log_probs(numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                 s_mid_fr_mask, \
                                                 numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                 s_source_en_mask) * (1-alpha)
                #print "time to calculate reward: ", time.time()-u1
                #print reward
                u1 = time.time()
                cost_en_fr = f_dual_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask, reward_en_fr)
                if numpy.isnan(cost_en_fr):
                    ipdb.set_trace()
                #print "time to calculate cost :", time.time()-u1
                u1 = time.time()
                f_dual_update_en_fr(lrate)
                #print "time to dual update :", time.time()-u1
                u1 = time.time()
                if optional_update:
                    cost_en_fr = f_dual_second_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask, reward_en_fr)
                    f_dual_second_update_en_fr(lrate)
                    
            #play fr --> en:
            if x_fr_s is not None:
                s_source_fr = []
                s_mid_en = []
                s_mid_en_2 = []
                reward = []
                u1 = time.time()

                for jj in xrange(x_fr_s.shape[1]):
                    stochastic = True
                    x_current = x_fr_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_fr.astype('int64')[:, jj].sum(),:]
                    #sampling
                    sample, score, sample_word_probs, alignment, hyp_graph = nmt_fr_en.gen_sample(
                                           x_current,
                                           k=12,
                                           maxlen=30,
                                           stochastic=stochastic,
                                           argmax=False,
                                           suppress_unk=True,
                                           return_hyp_graph=False)
                    tmp = []
                    for xs in x_fr[jj]:
                        tmp.append([xs])
                    for ss in sample:
                        s_mid_en.append(ss)
                        s_mid_en_2.append(ss)
                        s_source_fr.append(tmp)
                s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask = prepare_data_bi(s_source_fr, s_mid_en)
                s_mid_en_2, s_mid_en_2_mask = prepare_data_mono(s_mid_en_2)
                #print "time sampling one batch:", time.time() - u1
                u1 = time.time()                                                                
                #Time for dual ascent update: average over batch then over samples
                reward_fr_en = lm_en.f_log_probs(s_mid_en_2, s_mid_en_2_mask) * alpha\
                                                + nmt_en_fr.f_log_probs(numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                 s_mid_en_mask, \
                                                 numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                 s_source_fr_mask) * (1-alpha)
                #print "time to calculate reward: ", time.time() - u1                                                                        
                #print reward
                u1 = time.time()
                cost_fr_en = f_dual_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask, reward_fr_en)
                if numpy.isnan(cost_fr_en):
                    ipdb.set_trace()
                #print "time to calculate cost :", time.time()-u1
                u1 = time.time()
                f_dual_update_fr_en(lrate)
                #print "time to dual update :", time.time()-u1
                if optional_update:
                    cost_fr_en = f_dual_second_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask, reward_fr_en)
                    f_dual_second_update_fr_en(lrate)
        #Standard-using bilingual setence pair update 
            #update en->fr model's parameters
            u1 = time.time()

            if x_en_en_fr is not None:
                cost_ce_en_fr = f_grad_shared_en_fr(x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr)
                
                # do the update on parameters
                f_update_en_fr(lrate_bi)
            
            if x_fr_fr_en is not None:
                cost_ce_fr_en = f_grad_shared_fr_en(x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en)
                # do the update on parameters
                f_update_fr_en(lrate_bi)
            #print "time to standard update :", time.time()-u1
                                                        
        #print
            if numpy.mod(uidx,dispFreq) ==0:
                ud = time.time()-ud_start
                ud_start = time.time()
                print 'epoch:', i , 'Update: ', uidx,\
                "cost_en_fr: %f cost_fr_en: %f" % (cost_en_fr,cost_fr_en),\
                "cost_ce_en_fr: %f cost_ce_fr_en: %f" % (cost_ce_en_fr,cost_ce_fr_en),\
                'UD: ', ud
                
                saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)

                both_params = dict(theano_util.unzip_from_theano(tparams_en_fr))
                numpy.savez(saveto_uidx, **both_params)
                
            #load for next batch:   
                    
            x_en = train_en.next()
            x_en_s, x_mask_en = prepare_data_mono(x_en, maxlen=maxlen,
                                                        n_words=n_words)
            x_fr = train_fr.next()
            x_fr_s, x_mask_fr = prepare_data_mono(x_fr, maxlen=maxlen,
                                                        n_words=n_words)
                                                        
            x_en_en_fr, x_fr_en_fr = train_en_fr.next()
            x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = prepare_data_bi(x_en_en_fr, x_fr_en_fr,maxlen=maxlen,
                                                        n_words=n_words)
            x_fr_fr_en, x_en_fr_en = train_fr_en.next()
            x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = prepare_data_bi(x_fr_fr_en, x_en_fr_en,maxlen=maxlen,
                                                        n_words=n_words)
            
  
    return 0

train()
