#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import LM
from LM import lm
from nematus import nmt,theano_util,data_iterator,util,optimizers, training_progress
import theano
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

profile = False

home_dir = "/users/limsi_nmt/minhquang/"

def BP(source,candidate):
    l1 = len(source)
    l2 = len(candidate)
    if l2 > l1:
        return 1
    return numpy.exp(1-float(l1)/float(l2))    

def length_constraint(seq1,seq2):
    return abs(len(seq1)-len(seq2))/float(len(seq1)+len(seq2))

def init_theano_params(params,target):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk, target = target)
    return tparams

def dist_to_cluster(vec,cluster):
    dist = [numpy.linalg.norm(vec - vec1) for vec1 in cluster]
    return min(dist)

def dist_cluster_2_cluster(cluster1, cluster2):
    if not cluster1 or not cluster2:
        return 3.0
    dist1 = [dist_to_cluster(vec, cluster2) for vec in cluster1]
    dist2 = [dist_to_cluster(vec, cluster1) for vec in cluster2]
    return min(sum(dist1)/float(len(dist1)),sum(dist2)/float(len(dist2)))

def dist_bag_of_words_cluster_2_cluster(word_emb_en, word_emb_fr, worddict_en_r, worddict_fr_r, seq_en, seq_fr):
    cluster_en = []
    cluster_fr = []
    for ind in seq_en:
        if ind in worddict_en_r.keys():
            w = worddict_en_r[ind]        
            if w in word_emb_en.keys() :
                cluster_en.append(word_emb_en[w])
    for ind in seq_fr:
        if ind in worddict_fr_r.keys():        
            w = worddict_fr_r[ind]        
            if w in word_emb_fr.keys() :
                cluster_fr.append(word_emb_fr[w])
    return dist_cluster_2_cluster(cluster_en,cluster_fr)
    
def dist_bag_of_words_center_2_center(word_emb_en, word_emb_fr, worddict_en_r, worddict_fr_r, seq_en, seq_fr):
    dim = len(word_emb_en[worddict_en_r[seq_en[0]]])
    center_en = numpy.zeros(dim)
    center_fr = numpy.zeros(dim)
    c = 0
    for ind in seq_en:
        if ind in worddict_en_r.keys():
            w = worddict_en_r[ind]        
            if w in word_emb_en.keys() :
                center_en = center_en + word_emb_en[w]
                c = c + 1
    center_en = center_en / float(c)
    empty1 = c
    c = 0
    for ind in seq_fr:
        if ind in worddict_fr_r.keys():
            w = worddict_fr_r[ind]
            if w in word_emb_fr.keys():
                center_fr = center_fr + word_emb_fr[w]
                c = c + 1        
    center_fr = center_fr / float(c)
    empty2 = c
    if empty1 == 0 or empty2 == 0 :
        return 3.0
    return numpy.linalg.norm(center_en - center_fr)

def de_factor(seqs):
    new = []
    for seq in seqs:
            ss = []
            for s in seq:
                ss.append(s[0])
            new.append(ss)
    return new    

def dual_second_ascent(lr1, lr2, alpha, tparams_1, tparams_2, grads_1,\
                       grads_2, inps_1, inps_2, reward, avg_reward, source, target):     
    
    g_shared_1 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_forward_grad_second_shared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    g_shared_2 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_backward_grad_second_shared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    g_up_1 = [(g1, g2) for g1,g2 in zip(g_shared_1,grads_1)]
    g_up_2 = [(g1, -g2) for g1,g2 in zip(g_shared_2,grads_2)]
    
    f_grad_second_shared = theano.function(inps_1 + inps_2 + [reward], avg_reward, updates = g_up_1 + g_up_2, on_unused_input='ignore')
    
    params_up_1 = [(p , p + lr1 * g) for p,g in zip(theano_util.itemlist(tparams_1), g_shared_1)]
    params_up_2 = [(p , p + lr2 * (1-alpha) * g) for p,g in zip(theano_util.itemlist(tparams_2), g_shared_2)]
    
    f_second_update = theano.function([lr1,lr2], [], updates = params_up_1 + params_up_2, on_unused_input='ignore')
    
    return f_grad_second_shared, f_second_update

def dual_ascent(lr, tparams, grads, inps, reward, avg_reward, direction):     
    
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_forward_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
        
    f_grad_shared = theano.function(inps + [reward], avg_reward, updates = g_up, on_unused_input='ignore')
    
    params_up = [(p , p + lr * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    
    f_update = theano.function([lr], [], updates = params_up, on_unused_input='ignore')
    
    return f_grad_shared, f_update

def adadelta_dual_ascent(lr, tparams, grads, inps, reward, avg_reward, direction):
    g_shared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_forward_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    g_squared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_forward_grad_squared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    x_squared = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_forward_delta_squared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
       
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
    g_acc_up = [(g1, 0.95 * g1 + 0.05 * (g2 ** 2)) for g1,g2 in zip(g_squared,g_shared)]
    
    f_grad_shared = theano.function(inps + [reward], avg_reward, updates = g_up + g_acc_up, on_unused_input='ignore')

    updir = [(T.sqrt(delta_x_s + 1e-6) / T.sqrt(g_s + 1e-6) * g) \
             for delta_x_s,g_s,g in zip(x_squared,g_squared,g_shared)]
    
    delta_x_acc_up = [(delta_x1, 0.95 * delta_x1 + 0.05 * (delta_x ** 2)) \
                      for delta_x1,delta_x in zip(x_squared, updir)]        
    
    params_up = [(p , p + g) for p,g in zip(theano_util.itemlist(tparams), updir)]
    
    f_update = theano.function([lr], [], updates = delta_x_acc_up + params_up, on_unused_input='ignore')
    
    return f_grad_shared, f_update

def adadelta_second_dual_ascent(lr1, lr2, alpha, tparams_1, tparams_2, grads_1,\
                         grads_2, inps_1, inps_2, reward, avg_reward, source, target):
    g_shared_1 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_forward_grad_second_shared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    g_squared_1 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_forward_grad_second_squared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    x_squared_1 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_forward_delta_second_squared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    
    g_shared_2 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_backward_grad_second_shared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    g_squared_2 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_backward_grad_second_squared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    x_squared_2 = [ theano.shared(p.get_value()*numpy.float32(0.),name= '%s_%s_%s_backward_delta_second_squared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ] 
    
    g_up_1 = [(g1, g2) for g1,g2 in zip(g_shared_1,grads_1)]
    g_up_2 = [(g1, -g2) for g1,g2 in zip(g_shared_2,grads_2)]
    g_acc_up_1 = [(g1, 0.95 * g1 + 0.05 * (g2 ** 2)) for g1,g2 in zip(g_squared_1,g_shared_1)]
    g_acc_up_2 = [(g1, 0.95 * g1 + 0.05 * (g2 ** 2)) for g1,g2 in zip(g_squared_2,g_shared_2)]

    f_grad_second_shared = theano.function(inps_1 + inps_2 + [reward], avg_reward, updates = g_up_1 + g_up_2 + g_acc_up_1 + g_acc_up_2, on_unused_input='ignore')
    
    updir_1 = [(T.sqrt(delta_x_s + 1e-6) / T.sqrt(g_s + 1e-6) * g) \
             for delta_x_s,g_s,g in zip(x_squared_1,g_squared_1,g_shared_1)]
    
    updir_2 = [(T.sqrt(delta_x_s + 1e-6) / T.sqrt(g_s + 1e-6) * g) \
             for delta_x_s,g_s,g in zip(x_squared_2,g_squared_2,g_shared_2)] 
    
    delta_x_acc_up_1 = [(delta_x1, 0.95 * delta_x1 + 0.05 * (delta_x ** 2)) \
                      for delta_x1,delta_x in zip(x_squared_1, updir_1)]  
    delta_x_acc_up_2 = [(delta_x1, 0.95 * delta_x1 + 0.05 * (delta_x ** 2)) \
                      for delta_x1,delta_x in zip(x_squared_2, updir_2)]  
    
    params_up_1 = [(p , p + g) for p,g in zip(theano_util.itemlist(tparams_1), updir_1)]
    params_up_2 = [(p , p + (1-alpha) * g) for p,g in zip(theano_util.itemlist(tparams_2), updir_2)]
    
    f_second_update = theano.function([lr1,lr2], [], updates = params_up_1 + params_up_2 + delta_x_acc_up_1 + delta_x_acc_up_2 , on_unused_input='ignore')
    
    return f_grad_second_shared, f_second_update

def load_word_emb_bilbowa(word_emb_en, word_emb_fr, word_emb_en_path, word_emb_fr_path):
       
    with open(word_emb_en_path,"r") as file_en:
        for l in file_en:
            line= l.split()
            vec = numpy.array([numpy.float32(ii) for ii in line[1:]])
            word_emb_en[line[0]] = vec
            
    with open(word_emb_fr_path,"r") as file_fr:
        for l in file_fr:
            line= l.split()
            vec = numpy.array([numpy.float32(ii) for ii in line[1:]])
            word_emb_fr[line[0]] = vec  
    
    return word_emb_en, word_emb_fr


def train(dim_word = 512,  # word vector dimensionality
              dim = 1024,  # the number of LSTM units
              factors = 1, # input factors
              dim_per_factor = None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
              encoder = 'gru',
              decoder = 'gru_cond',
              lrate_fw = 0.0001,  # learning rate
              lrate_bw = 0.001,
              lrate_bi = 0.0001,
              print_gradient = True,
              n_words_en = 15000,  # english vocabulary size
              n_words_fr = 15000 ,  # french vocabulary size
              optimizers_ = None,
              optimizers_biling = "sgd",
              maxlen=30,  # maximum length of the description
              disp_grad_Freq = 400,
              dispFreq = 100,
              saveFreq = 400,
              validFreq = 2000,
              batch_size= 30,
              valid_batch_size = 60,
              save = True,
              warm_start_ = True,
              saveto = home_dir + 'Dual_NMT/models/dual2/model_dual.npz',
              use_dropout = False,
              use_second_update = True,
              using_word_emb_Bilbowa = False,
              dropout_embedding = 0.2, # dropout for input embeddings (0: no dropout)
              dropout_hidden = 0.2, # dropout for hidden layers (0: no dropout)
              dropout_source = 0, # dropout source words (0: no dropout)
              dropout_target = 0, # dropout target words (0: no dropout)
              reload_ = False,
              tie_encoder_decoder_embeddings = False, # Tie the input embeddings of the encoder and the decoder (first factor only)
              tie_decoder_embeddings = False, # Tie the input embeddings of the decoder with the softmax output embeddings
              encoder_truncate_gradient = -1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
              decoder_truncate_gradient = -1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
              alpha = 0.005,
              reward_scale = 1.0,
              clip_c = 1.,
              external_validation_script_en_fr = None,
	      args_en_fr = "",
              external_validation_script_fr_en = None,
	      args_fr_en = "",
              beam_search_size = 2,
              dist_scale = 10.0,
              length_constraint_scale = 10.0,
              distance = "dist_bag_of_words_cluster_2_cluster",
              length_constraint_ = "length_constraint",
              valid_en = home_dir + "Dual_NMT/data/validation/hit/hit.en.tok.shuf.dev.tok",
              valid_fr = home_dir + "Dual_NMT/data/validation/hit/hit.fr.tok.shuf.dev.tok",
              dataset_bi_en = home_dir + "Dual_NMT/data/train/train10/train10.en.tok",
              dataset_bi_fr = home_dir + "Dual_NMT/data/train/train10/train10.fr.tok",
              dataset_mono_en = home_dir + "Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok",
              dataset_mono_fr = home_dir + "Dual_NMT/data/train/hit/hit.fr.tok.shuf.train.tok",
              vocab_en = home_dir + "Dual_NMT/data/vocabulaires/concatenated.en.tok.json",
              vocab_fr = home_dir + "Dual_NMT/data/vocabulaires/concatenated.fr.tok.json",
              test_en = home_dir + "Dual_NMT/data/validation/hit/hit.en.tok.shuf.test.tok",
              test_fr = home_dir + "Dual_NMT/data/validation/hit/hit.fr.tok.shuf.test.tok",
              path_trans_en_fr = home_dir + "Dual_NMT/models/NMT/lowercased/model_en_fr.npz.npz.best_bleu",
              path_trans_fr_en = home_dir + "Dual_NMT/models/NMT/lowercased/model_fr_en.npz.npz.best_bleu",
              path_mono_en = home_dir + "Dual_NMT/models/LM/lowercased/model_lm_en.npz",
              path_mono_fr = home_dir + "Dual_NMT/models/LM/lowercased/model_lm_fr.npz",
              word_emb_en_path = home_dir + "Dual_NMT/data/word_emb/envec.txt",
              word_emb_fr_path = home_dir + "Dual_NMT/data/word_emb/frvec.txt",
              reload_training_progress = False              
        ):
    
    #ipdb.set_trace()
    print "saveto:",saveto
    #stats:
    stats = dict()
    stats["cost_avg_en_fr"] = [] 
    stats["cost_avg_fr_en"] = []
    stats["cost_ce_avg_en_fr"] = []
    stats["cost_ce_avg_fr_en"] = []

    stats["dist_avg_en_fr"] = []
    stats["forward_avg_en_fr"] = [] 
    stats["backward_avg_en_fr"] = [] 
    stats["length_diff_avg_en_fr"] = []

    stats["dist_avg_fr_en"] = []
    stats["forward_avg_fr_en"] = []
    stats["backward_avg_fr_en"] = [] 
    stats["length_diff_avg_fr_en"] = []

    # Model options
    u = time.time()
    u1 = time.time()
    model_options_trans = OrderedDict(sorted(locals().copy().items()))
    model_options_en = OrderedDict()
    model_options_fr = OrderedDict()
    

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
    
    model_options_fr_en["datasets"] = [dataset_bi_fr,dataset_bi_en]
    model_options_fr_en["dictionaries"] = [vocab_fr,vocab_en]
    model_options_fr_en["n_words_src"] = n_words_fr
    model_options_fr_en["n_words"] = n_words_en
                       
    model_options_en_fr["datasets"] = [dataset_bi_en,dataset_bi_fr]
    model_options_en_fr["dictionaries"] = [vocab_en,vocab_fr]  
    model_options_en_fr["n_words_src"] = n_words_en
    model_options_en_fr["n_words"] = n_words_fr
    json.dump(dict(model_options_en_fr),open('%s.model_options_en_fr.npz.json'%saveto,'wb'),indent=2)
    json.dump(dict(model_options_fr_en),open('%s.model_options_fr_en.npz.json'%saveto,'wb'),indent=2)
        
    #training progression file:
    best_p_en_fr = None
    best_p_fr_en = None
    best_opt_p_en_fr = None
    best_opt_p_fr_en = None
    training_progress_en_fr = training_progress.TrainingProgress()
    training_progress_en_fr.uidx = 0
    training_progress_en_fr.eidx = 0
    training_progress_en_fr.history_errs = []
    training_progress_file_en_fr = saveto + '.en_fr.progress.json'
    
    training_progress_fr_en = training_progress.TrainingProgress()
    training_progress_fr_en.uidx = 0
    training_progress_fr_en.eidx = 0
    training_progress_fr_en.history_errs = []
    training_progress_file_fr_en = saveto + '.fr_en.progress.json'
    
    print "Loading options time:", time.time()-u1
    u1 = time.time()
    #hyperparameters:
    alp = theano.shared(numpy.float32(alpha),name="alpha")
    
    # Translation Model:
    #load dictionary
        #English:
    worddict_en_r = dict()
    worddict_fr_r = dict()
    worddict_en = util.load_dict(vocab_en)
    for kk,vv in worddict_en.iteritems():
        worddict_en_r[vv] = kk
        #French:
    worddict_fr = util.load_dict(vocab_fr)
    for kk,vv in worddict_fr.iteritems():
        worddict_fr_r[vv] = kk
                   
    # Intilize params and tparams
    params_en_fr = nmt.init_params(model_options_en_fr)
    params_fr_en = nmt.init_params(model_options_fr_en)
        #reload
    #en->fr:
    if warm_start_ and os.path.exists(path_trans_en_fr) and not reload_:
        print 'Reloading en-fr warm start model parameters'
        params_en_fr = theano_util.load_params(path_trans_en_fr, params_en_fr)
    #fr->en:
    if warm_start_ and os.path.exists(path_trans_fr_en) and not reload_:
        print 'Reloading fr-en warm start model parameters'
        params_fr_en = theano_util.load_params(path_trans_fr_en, params_fr_en)
        
    if reload_:        
        saveto_en_fr = '{}.en_fr.npz'.format(
                        os.path.splitext(saveto)[0])         
        if os.path.exists(saveto_en_fr):
            print 'Reloading en_fr model parameters'
            params_en_fr = theano_util.load_params(saveto_en_fr, params_en_fr)
            
        saveto_fr_en = '{}.fr_en.npz'.format(
                        os.path.splitext(saveto)[0])         
        if os.path.exists(saveto_fr_en):
            print 'Reloading fr_en model parameters'
            params_fr_en = theano_util.load_params(saveto_fr_en, params_fr_en)
    
    tparams_en_fr = theano_util.init_theano_params(params_en_fr)
    tparams_fr_en = theano_util.init_theano_params(params_fr_en)
    print "Loading parameters time:", time.time()-u1
    u1 = time.time()
    # build models
    print "build nmt models ... ",
    trng_en_fr, use_noise_en_fr, x_en, \
    x_en_mask, y_fr, y_fr_mask, opt_ret_en_fr, cost_en_fr = nmt.build_model(tparams_en_fr,model_options_en_fr)
    
    trng_fr_en, use_noise_fr_en, x_fr, \
    x_fr_mask, y_en, y_en_mask, opt_ret_fr_en, cost_fr_en = nmt.build_model(tparams_fr_en,model_options_fr_en)
    
    inps_en_fr = [x_en, x_en_mask, y_fr, y_fr_mask]
    inps_fr_en = [x_fr, x_fr_mask, y_en, y_en_mask]
    print "Done \n"   
    
    #build samplers
    print "Build samplers ...",
    f_init_en_fr, f_next_en_fr = nmt.build_sampler(tparams_en_fr, model_options_en_fr, use_noise_en_fr, trng_en_fr)
    f_init_fr_en, f_next_fr_en = nmt.build_sampler(tparams_fr_en, model_options_fr_en, use_noise_fr_en, trng_fr_en)
    print "Done\n"
    
    #build g_log_probs
    f_log_probs_en_fr = theano.function(inps_en_fr, -cost_en_fr)
    f_log_probs_fr_en = theano.function(inps_fr_en, -cost_fr_en)
    
    print "Loading translation models time:", time.time()-u1
    u1 = time.time()
    # Compute gradient
    print "Build gradient ...",
    
        #rewards and avg_reward
    reward_en_fr = T.vector("reward_en_fr")
    reward_fr_en = T.vector("reward_fr_en")
    
    avg_reward_en_fr = T.mean(reward_en_fr)
    avg_reward_fr_en = T.mean(reward_fr_en)


        # -cost = log(p(s_mid|s))
    new_cost_en_fr = T.mean(reward_en_fr * (- cost_en_fr))
    new_cost_fr_en = T.mean(reward_fr_en * (- cost_fr_en))
        # cross entropy
    cost_ce_en_fr = cost_en_fr.mean()
    cost_ce_fr_en = cost_fr_en.mean()    

        # gradient newcost = gradient( avg reward * -cost) = avg reward_i * gradient( -cost_i) = avg reward_i * gradient(log p(s_mid | s)) stochastic approximation of policy gradient
    grad_en_fr = T.grad(new_cost_en_fr, wrt = theano_util.itemlist(tparams_en_fr)) 
    grad_fr_en = T.grad(new_cost_fr_en, wrt = theano_util.itemlist(tparams_fr_en)) 
    
    grad_ce_en_fr = T.grad(cost_ce_en_fr, wrt = theano_util.itemlist(tparams_en_fr))
    grad_ce_fr_en = T.grad(cost_ce_fr_en, wrt = theano_util.itemlist(tparams_fr_en))
    
    # apply gradient clipping here
    if clip_c > 0.:
        g1 = 0.
        for g in grad_en_fr:
            g1 += (g**2).sum()
        new_grads_1 = []
        for g in grad_en_fr:
            new_grads_1.append(T.switch(g1 > (clip_c**2),
                            g / T.sqrt(g1) * clip_c, g))
        grad_en_fr = new_grads_1
        
        g2 = 0.
        for g in grad_fr_en:
            g2 += (g**2).sum()
        new_grads_2 = []
        for g in grad_fr_en:
            new_grads_2.append(T.switch(g2 > (clip_c**2),
                            g / T.sqrt(g2) * clip_c, g))
        grad_fr_en = new_grads_2
        
        g3 = 0.
        for g in grad_ce_en_fr:
            g3 += (g**2).sum()
        new_grads_3 = []
        for g in grad_ce_en_fr:
            new_grads_3.append(T.switch(g3 > (clip_c**2),
                            g / T.sqrt(g3) * clip_c, g))
        grad_ce_en_fr = new_grads_3
        
        g4 = 0.
        for g in grad_ce_fr_en:
            g4 += (g**2).sum()
        new_grads_4 = []
        for g in grad_ce_fr_en:
            new_grads_4.append(T.switch(g4 > (clip_c**2),
                            g / T.sqrt(g4) * clip_c, g))
        grad_ce_fr_en = new_grads_4
    """
    #function to get value of gradients
    g_en_fr = theano.function(inps_en_fr + [reward_en_fr], grad_en_fr)
    g_fr_en = theano.function(inps_fr_en + [reward_fr_en], grad_fr_en)
    
    g_ce_en_fr = theano.function(inps_en_fr, grad_ce_en_fr)
    g_ce_fr_en = theano.function(inps_fr_en, grad_ce_fr_en)
    """
        #build f_grad_shared: average rewards, f_update: update params by gradient newcost
    lr_forward = T.scalar('lrate_forward')
    lr_backward = T.scalar('lrate_backward')
    lr1 = T.scalar('lrate1')
    lr2 = T.scalar('lrate2')
    if optimizers_ is not None:
        f_dual_grad_shared_en_fr, f_dual_update_en_fr = eval("%s_dual_ascent"%optimizers_)(lr_forward, tparams_en_fr, grad_en_fr, \
                                                                inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr" ) 
        f_dual_grad_shared_fr_en, f_dual_update_fr_en = eval("%s_dual_ascent"%optimizers_)(lr_forward, tparams_fr_en, grad_fr_en, \
                                                                inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
    else:    
        f_dual_grad_shared_en_fr, f_dual_update_en_fr = dual_ascent(lr_forward, tparams_en_fr, grad_en_fr, \
                                                                    inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr" ) 
        f_dual_grad_shared_fr_en, f_dual_update_fr_en = dual_ascent(lr_forward, tparams_fr_en, grad_fr_en, \
                                                                    inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
    
    if use_second_update:
        if optimizers_ is not None:
            f_dual_grad_shared_en_fr, f_dual_update_en_fr = eval("%s_second_dual_ascent"%optimizers_)(lr_forward, lr_backward,\
                                                                                         alp, tparams_en_fr,\
                                                                                         tparams_fr_en, grad_en_fr,\
                                                                                         grad_ce_fr_en, inps_en_fr, inps_fr_en,\
                                                                                         reward_en_fr, avg_reward_en_fr, "en", "fr")
            f_dual_grad_shared_fr_en, f_dual_update_fr_en = eval("%s_second_dual_ascent"%optimizers_)(lr_forward, lr_backward,\
                                                                                         alp, tparams_fr_en,\
                                                                                         tparams_en_fr, grad_fr_en,\
                                                                                         grad_ce_en_fr, inps_fr_en, inps_en_fr,\
                                                                                         reward_fr_en, avg_reward_fr_en, "fr", "en") 
        
        else:
            f_dual_grad_shared_en_fr, f_dual_update_en_fr = dual_second_ascent(lr_forward, lr_backward, alp, tparams_en_fr,\
                                                                                             tparams_fr_en, grad_en_fr,\
                                                                                             grad_ce_fr_en, inps_en_fr, inps_fr_en,\
                                                                                             reward_en_fr, avg_reward_en_fr, "en", "fr")
            f_dual_grad_shared_fr_en, f_dual_update_fr_en = dual_second_ascent(lr_forward, lr_backward, alp, tparams_fr_en,\
                                                                                             tparams_en_fr, grad_fr_en,\
                                                                                             grad_ce_en_fr, inps_fr_en, inps_en_fr,\
                                                                                             reward_fr_en, avg_reward_fr_en, "fr", "en") 
        
    f_grad_shared_en_fr, f_update_en_fr, _ = eval('optimizers.%s'%optimizers_biling)(lr1, tparams_en_fr, grad_ce_en_fr, inps_en_fr, cost_ce_en_fr)
    f_grad_shared_fr_en, f_update_fr_en, _ = eval('optimizers.%s'%optimizers_biling)(lr2, tparams_fr_en, grad_ce_fr_en, inps_fr_en, cost_ce_fr_en)
    
    print "Done\n"
    print "Building gradients time:", time.time()-u1
    u1 = time.time()
    #build language model
    model_options_en['encoder'] = 'gru'
    model_options_en['dim'] = 1024
    model_options_en['dim_word'] = 512
    model_options_en['n_words'] = n_words_en
    model_options_en['dataset'] = dataset_mono_en
    
    model_options_fr['encoder'] = 'gru'
    model_options_fr['dim'] = 1024
    model_options_fr['dim_word'] = 512
    model_options_fr['n_words'] = n_words_fr
    model_options_fr['dataset'] = dataset_mono_fr

    print "Build language models ...",
    params_en = lm.init_params(model_options_en)
    params_fr = lm.init_params(model_options_fr)
    json.dump(dict(model_options_en),open("%s.model_options_en.npz.json"%saveto,"wb"))
    json.dump(dict(model_options_fr),open("%s.model_options_fr.npz.json"%saveto,"wb"))

    # reload parameters
    if reload_ and os.path.exists(path_mono_en):
        params_en = lm.load_params(path_mono_en, params_en)
    tparams_en = lm.init_tparams(params_en)
    if reload_ and os.path.exists(path_mono_fr):
        params_fr = lm.load_params(path_mono_fr, params_fr)
    tparams_fr = lm.init_tparams(params_fr)
    
    # build the language models
    trng_en, use_noise_en, x_en, x_mask_en, opt_ret_en, cost_en = lm.build_model(tparams_en, model_options_en)
    inps_en = [x_en, x_mask_en]
    trng_fr, use_noise_fr, x_fr, x_mask_fr, opt_ret_fr, cost_fr = lm.build_model(tparams_fr, model_options_fr)
    inps_fr = [x_fr, x_mask_fr]
    f_log_probs_en = theano.function(inps_en, -cost_en, profile=profile)
    f_log_probs_fr = theano.function(inps_fr, -cost_fr, profile=profile)

    print "Done"	
    print "Loading language models and theirs parameters time:", time.time()-u1
    print "Compilation time:", time.time()-u 
    
    #load word embeddings:
        
    #load 
    word_emb_en = dict()
    word_emb_fr = dict()
    if using_word_emb_Bilbowa:
        word_emb_en, word_emb_fr = load_word_emb_bilbowa(word_emb_en, word_emb_fr, word_emb_en_path, word_emb_fr_path)
        print "word embedding loaded"
    
    #Soft-landing phrase   
    
    max_epochs = 500
    c_fb_batches_en_fr = 0
    c_d_batches_en_fr = 0
    c_fb_batches_fr_en = 0
    c_d_batches_fr_en = 0
    cost_acc_en_fr = 0
    cost_ce_acc_en_fr = 0
    cost_acc_fr_en = 0
    cost_ce_acc_fr_en = 0

    dist_acc_en_fr = 0
    forward_acc_en_fr = 0
    backward_acc_en_fr = 0    
    length_diff_acc_en_fr = 0

    dist_acc_fr_en = 0
    forward_acc_fr_en = 0
    backward_acc_fr_en = 0
    length_diff_acc_fr_en = 0

    ud_start = time.time()
    p_validation_en_fr = None
    p_validation_fr_en = None
    
    # validation sets:

    valid_en_fr = data_iterator.TextIterator(valid_en, valid_fr,\
                     [vocab_en], vocab_fr,\
                     batch_size=batch_size * 2,\
                     maxlen=maxlen,\
                     n_words_source=n_words_en,\
                     n_words_target=n_words_fr)

    valid_fr_en = data_iterator.TextIterator(valid_fr, valid_en,\
                     [vocab_fr], vocab_en,\
                     batch_size=batch_size * 2,\
                     maxlen=maxlen,\
                     n_words_source=n_words_fr,\
                     n_words_target=n_words_en)
    
    for training_progress_en_fr.eidx in xrange(training_progress_en_fr.eidx, max_epochs):  
        training_progress_fr_en.eidx = training_progress_en_fr.eidx
        
        train_en = LM.data_iterator.TextIterator(dataset_mono_en, vocab_en, batch_size = batch_size /2,\
                                                 maxlen = maxlen, \
                                                 n_words_source = n_words_en)
        train_fr = LM.data_iterator.TextIterator(dataset_mono_fr, vocab_fr, batch_size = batch_size /2,\
                                                 maxlen = maxlen, \
                                                 n_words_source = n_words_fr)
        train_en_fr = data_iterator.TextIterator(dataset_bi_en, dataset_bi_fr,\
                     [vocab_en], vocab_fr,\
                     batch_size = batch_size /2,\
                     maxlen = maxlen,\
                     n_words_source = n_words_en,\
                     n_words_target = n_words_fr, shuffle_each_epoch = True)

        train_fr_en = data_iterator.TextIterator(dataset_bi_fr, dataset_bi_en,\
                     [vocab_fr], vocab_en,\
                     batch_size = batch_size /2,\
                     maxlen = maxlen,\
                     n_words_source = n_words_fr,\
                     n_words_target = n_words_en, shuffle_each_epoch = True)
                         
        x_en = train_en.next()
        x_en_s, x_mask_en = lm.prepare_data(x_en, maxlen = maxlen,
                                                            n_words = n_words_en)
        x_fr = train_fr.next()
        x_fr_s, x_mask_fr = lm.prepare_data(x_fr, maxlen = maxlen,
                                                            n_words = n_words_fr)
        
        x_en_en_fr, x_fr_en_fr = train_en_fr.next()
        x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr, maxlen = maxlen,
                                                            n_words = n_words_en)
        x_fr_fr_en, x_en_fr_en = train_fr_en.next()
        x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en, maxlen = maxlen,
                                                            n_words = n_words_fr)
        while x_en_s is not None or x_fr_s is not None:
            training_progress_en_fr.uidx += 1
            training_progress_fr_en.uidx += 1
       
        #Dual update
            # play game en->fr:
            
            if x_en_s is not None:
                c_fb_batches_en_fr += 1
                s_source_en = []
                s_mid_fr = []
                s_mid_fr_2 = []
                u1 = time.time()
                for jj in xrange(x_en_s.shape[1]):
                    stochastic = True
                    x_current = x_en_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_en.astype('int64')[:, jj].sum(),:]
                    #sampling
                    sample, score, sample_word_probs, alignment, hyp_graph = nmt.gen_sample([f_init_en_fr],\
                                           [f_next_en_fr],
                                           x_current,
                                           k=beam_search_size,
                                           maxlen=maxlen,
                                           stochastic=stochastic,
                                           argmax=False,
                                           suppress_unk=False,
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
                s_source_en_tmp = s_source_en
                s_mid_fr_tmp = s_mid_fr
                s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask = nmt.prepare_data(s_source_en, s_mid_fr)
                s_mid_fr_2, s_mid_fr_2_mask = lm.prepare_data(s_mid_fr_2)
                #print "time for prepare data: ", time.time() - u1
                #Time for dual ascent update: average over batch then over samples
                u1 = time.time()
                length_constraint_en_fr = numpy.array([eval(length_constraint_)(seq1,seq2) \
                                                 for seq1,seq2 in zip(de_factor(s_source_en_tmp),s_mid_fr_tmp)]).astype("float32")
                dist_BCOW_en_fr = numpy.array([eval(distance)(word_emb_en, word_emb_fr, worddict_en_r, worddict_fr_r, seq1,seq2) \
                                                 for seq1,seq2 in zip(de_factor(s_source_en_tmp),s_mid_fr_tmp)]).astype("float32")
                forward_en_fr = f_log_probs_fr(s_mid_fr_2, s_mid_fr_2_mask)  
                backward_en_fr = f_log_probs_fr_en(numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                 s_mid_fr_mask, \
                                                 numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                 s_source_en_mask)
                """
                print dist_BCOW_en_fr
                print forward_en_fr
                print backward_en_fr
                """
                reward_en_fr = ( forward_en_fr * alpha \
                               + backward_en_fr * (1-alpha) - dist_BCOW_en_fr * dist_scale - length_constraint_en_fr * length_constraint_scale ) * reward_scale
                #print "time to calculate reward: ", time.time()-u1
                u1 = time.time()
                if use_second_update:
                    cost_en_fr = f_dual_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask,\
                                                          numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                          s_mid_fr_mask, \
                                                          numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                          s_source_en_mask, reward_en_fr)
                    f_dual_update_en_fr(lrate_fw,lrate_bw)
                else:
                    cost_en_fr = f_dual_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask, reward_en_fr)
                    f_dual_update_en_fr(lrate_fw)

                cost_acc_en_fr += cost_en_fr
                dist_acc_en_fr += numpy.mean(dist_BCOW_en_fr)
                forward_acc_en_fr += numpy.mean(forward_en_fr)
                backward_acc_en_fr += numpy.mean(backward_en_fr)
                length_diff_acc_en_fr += numpy.mean(length_constraint_en_fr)

                """
                if print_gradient and numpy.mod(training_progress_en_fr.uidx,disp_grad_Freq)==0:
                    save_grad = g_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask, reward_en_fr)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                    save_grad = g_ce_fr_en(numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                              s_mid_fr_mask, \
                                                              numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                              s_source_en_mask)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                print "time to dual update :", time.time()-u1
                """
                if numpy.isnan(cost_en_fr):
                    ipdb.set_trace()
                    
            #play fr --> en:
            if x_fr_s is not None:
                c_fb_batches_fr_en += 1
                s_source_fr = []
                s_mid_en = []
                s_mid_en_2 = []
                u1 = time.time()

                for jj in xrange(x_fr_s.shape[1]):
                    stochastic = True
                    x_current = x_fr_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_fr.astype('int64')[:, jj].sum(),:]
                    #sampling
                    sample, score, sample_word_probs, alignment, hyp_graph = nmt.gen_sample([f_init_fr_en],\
                                           [f_next_fr_en],
                                           x_current,
                                           k=beam_search_size,
                                           maxlen=maxlen,
                                           stochastic=stochastic,
                                           argmax=False,
                                           suppress_unk=False,
                                           return_hyp_graph=False)
                    tmp = []
                    for xs in x_fr[jj]:
                        tmp.append([xs])
                    for ss in sample:
                        s_mid_en.append(ss)
                        s_mid_en_2.append(ss)
                        s_source_fr.append(tmp)
                s_source_fr_tmp = s_source_fr
                s_mid_en_tmp = s_mid_en
                s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask = nmt.prepare_data(s_source_fr, s_mid_en)
                s_mid_en_2, s_mid_en_2_mask = lm.prepare_data(s_mid_en_2)
                #print "time sampling one batch:", time.time() - u1
                u1 = time.time()  
                length_constraint_fr_en = numpy.array([eval(length_constraint_)(seq1,seq2) \
                                                 for seq1,seq2 in zip(s_mid_en_tmp,de_factor(s_source_fr_tmp))]).astype("float32")
                dist_BCOW_fr_en = numpy.array([eval(distance)(word_emb_en, word_emb_fr, worddict_en_r, worddict_fr_r, seq1,seq2) \
                                                 for seq1,seq2 in zip(s_mid_en_tmp,de_factor(s_source_fr_tmp))]).astype("float32")
                
                forward_fr_en = f_log_probs_en(s_mid_en_2, s_mid_en_2_mask)
                backward_fr_en = f_log_probs_en_fr(numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                 s_mid_en_mask, \
                                                 numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                 s_source_fr_mask)                
                """
                print dist_BCOW_fr_en                             
                print forward_fr_en                               
                print backward_fr_en
                """
                #Time for dual ascent update: average over batch then over samples
                reward_fr_en = ( forward_fr_en * alpha\
                               + backward_fr_en * (1-alpha) - dist_BCOW_fr_en * dist_scale - length_constraint_fr_en * length_constraint_scale ) * reward_scale
                #print "time to calculate reward: ", time.time() - u1                                                                        
                u1 = time.time()
                if use_second_update:
                    cost_fr_en = f_dual_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask,\
                                                          numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                          s_mid_en_mask, \
                                                          numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                          s_source_fr_mask, reward_fr_en)
                    f_dual_update_fr_en(lrate_fw,lrate_bw)
                else:
                    cost_fr_en = f_dual_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask, reward_fr_en)
                    f_dual_update_fr_en(lrate_fw)

                cost_acc_fr_en += cost_fr_en
                dist_acc_fr_en += numpy.mean(dist_BCOW_fr_en)
                forward_acc_fr_en += numpy.mean(forward_fr_en)
                backward_acc_fr_en += numpy.mean(backward_fr_en)
                length_diff_acc_fr_en += numpy.mean(length_constraint_fr_en)

                """
                if print_gradient and numpy.mod(training_progress_en_fr.uidx,disp_grad_Freq)==0:
                    save_grad = g_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask, reward_fr_en)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                    save_grad = g_ce_en_fr(numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                              s_mid_en_mask, \
                                                              numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                              s_source_fr_mask)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                """
                #print "time to dual update :", time.time()-u1                    
                if numpy.isnan(cost_fr_en):
                    ipdb.set_trace()
                        
        #Standard-using bilingual setence pair update 
            #update en->fr model's parameters
            u1 = time.time()
            if x_en_en_fr is not None:
                c_d_batches_en_fr += 1
                cost_ce_en_fr = f_grad_shared_en_fr(x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr)
                cost_ce_acc_en_fr += cost_ce_en_fr
                # do the update on parameters
                f_update_en_fr(lrate_bi)
                """
                if print_gradient and numpy.mod(training_progress_en_fr.uidx,disp_grad_Freq)==0:
                    save_grad = g_ce_en_fr(x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                """
                #print g_ce_en_fr(x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr)
            if x_fr_fr_en is not None:
                c_d_batches_fr_en += 1
                cost_ce_fr_en = f_grad_shared_fr_en(x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en)
                cost_ce_acc_fr_en += cost_ce_fr_en
                # do the update on parameters
                f_update_fr_en(lrate_bi)
                """
                if print_gradient and numpy.mod(training_progress_en_fr.uidx,disp_grad_Freq)==0:
                    save_grad = g_ce_fr_en(x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en)
                    print [save_grad[i].max() for i in range(len(save_grad))]
                """
                #print g_ce_fr_en(x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en)
            #print "time to standard update :", time.time()-u1
                                                           
        #print
            if numpy.mod(training_progress_en_fr.uidx, dispFreq) == 0:
                ud = time.time()-ud_start
                ud_start = time.time()
                cost_avg_en_fr = cost_acc_en_fr / float(c_fb_batches_en_fr)
                cost_avg_fr_en = cost_acc_fr_en / float(c_fb_batches_fr_en)

                cost_ce_avg_en_fr = cost_ce_acc_en_fr / float(c_d_batches_en_fr)
                cost_ce_avg_fr_en = cost_ce_acc_fr_en / float(c_d_batches_fr_en)

                dist_avg_en_fr = dist_acc_en_fr / float(c_fb_batches_en_fr)
                forward_avg_en_fr = forward_acc_en_fr / float(c_fb_batches_en_fr)
                backward_avg_en_fr = backward_acc_en_fr / float(c_fb_batches_en_fr)
                length_diff_avg_en_fr = length_diff_acc_en_fr / float(c_fb_batches_en_fr)
                
                dist_avg_fr_en = dist_acc_fr_en / float(c_fb_batches_fr_en)
                forward_avg_fr_en = forward_acc_fr_en / float(c_fb_batches_fr_en)
                backward_avg_fr_en = backward_acc_fr_en / float(c_fb_batches_fr_en)
                length_diff_avg_fr_en = length_diff_acc_fr_en / float(c_fb_batches_fr_en)
                
                #save stats
                stats["cost_avg_en_fr"].append(cost_avg_en_fr)
	        stats["cost_avg_fr_en"].append(cost_avg_fr_en)
    	        stats["cost_ce_avg_en_fr"].append(cost_ce_avg_en_fr)
    		stats["cost_ce_avg_fr_en"].append(cost_ce_avg_fr_en)
    
    		stats["dist_avg_en_fr"].append(dist_avg_en_fr)
    		stats["forward_avg_en_fr"].append(forward_avg_en_fr)
    		stats["backward_avg_en_fr"].append(backward_avg_en_fr)
    		stats["length_diff_avg_en_fr"].append(length_diff_avg_en_fr)

    		stats["dist_avg_fr_en"].append(dist_avg_fr_en)
    		stats["forward_avg_fr_en"].append(forward_avg_fr_en)
    		stats["backward_avg_fr_en"].append(backward_avg_fr_en)
    		stats["length_diff_avg_fr_en"].append(length_diff_avg_fr_en)


                print 'epoch:', training_progress_en_fr.eidx , 'Update: ', training_progress_en_fr.uidx,\
                "cost_en_fr: %f cost_fr_en: %f" % (cost_avg_en_fr, cost_avg_fr_en),\
                "cost_ce_en_fr: %f cost_ce_fr_en: %f" % (cost_ce_avg_en_fr, cost_ce_avg_fr_en),\
                "dist_avg_en_fr: %f dist_avg_fr_en: %f" % (dist_avg_en_fr, dist_avg_fr_en)
                print "forward_avg_en_fr: %f forward_avg_fr_en: %f" % (forward_avg_en_fr, forward_avg_fr_en),\
                "backward_avg_en_fr: %f backward_avg_fr_en: %f" % (backward_avg_en_fr, backward_avg_fr_en),\
                "length_diff_avg_en_fr: %f length_diff_avg_fr_en: %f" % (length_diff_avg_en_fr, length_diff_avg_fr_en),\
                'UD: ', ud
                
               	c_fb_batches_en_fr = 0
	        c_d_batches_en_fr = 0
    		c_fb_batches_fr_en = 0
    		c_d_batches_fr_en = 0

    		cost_acc_en_fr = 0
    		cost_ce_acc_en_fr = 0
    		cost_acc_fr_en = 0
    		cost_ce_acc_fr_en = 0

    		dist_acc_en_fr = 0
    		forward_acc_en_fr = 0
    		backward_acc_en_fr = 0
    		length_diff_acc_en_fr = 0

    		dist_acc_fr_en = 0
    		forward_acc_fr_en = 0
    		backward_acc_fr_en = 0
    		length_diff_acc_fr_en = 0

                
            if save and numpy.mod(training_progress_en_fr.uidx, saveFreq) == 0:
                saveto_uidx = '{}.iter{}.en_fr.npz'.format(
                        os.path.splitext(saveto)[0], training_progress_en_fr.uidx)

                both_params = dict(theano_util.unzip_from_theano(tparams_en_fr))
                numpy.savez(saveto_uidx, **both_params)
                
                saveto_uidx = '{}.iter{}.fr_en.npz'.format(
                        os.path.splitext(saveto)[0], training_progress_fr_en.uidx)

                both_params = dict(theano_util.unzip_from_theano(tparams_fr_en))
                numpy.savez(saveto_uidx, **both_params)    
                
            if save and numpy.mod(training_progress_en_fr.uidx, saveFreq) == 0:
                print 'Saving the best model_en_fr...',
                if best_p_en_fr is not None:
                    params_en_fr = best_p_en_fr
                else:
                    params_en_fr = theano_util.unzip_from_theano(tparams_en_fr, excluding_prefix='prior_')

                both_params = dict(theano_util.unzip_from_theano(tparams_en_fr))
                saveto_en_fr = '{}.en_fr.npz'.format(
                        os.path.splitext(saveto)[0])
                numpy.savez(saveto_en_fr, **both_params)
                training_progress_en_fr.save_to_json(training_progress_file_en_fr)
                print 'Done'
                
                print 'Saving the best model_fr_en...',
                if best_p_fr_en is not None:
                    params_fr_en = best_p_fr_en
                else:
                    params_fr_en = theano_util.unzip_from_theano(tparams_fr_en, excluding_prefix='prior_')

                both_params = dict(theano_util.unzip_from_theano(tparams_fr_en))
                saveto_fr_en = '{}.fr_en.npz'.format(
                        os.path.splitext(saveto)[0])
                numpy.savez(saveto_fr_en, **both_params)
                #print stats:
                saveto_stats = '{}.stats.npz'.format(
                        os.path.splitext(saveto)[0])
                numpy.savez(saveto_stats, **stats)
                training_progress_fr_en.save_to_json(training_progress_file_fr_en)
                print 'Done'

               
        # test on validation data:
            if numpy.mod(training_progress_en_fr.uidx, validFreq) == 0 or training_progress_en_fr.uidx == 1:
                use_noise_en_fr.set_value(0.)
                use_noise_fr_en.set_value(0.)
                
                valid_errs_en_fr, alignment = nmt.pred_probs(f_log_probs_en_fr, nmt.prepare_data,
                                        model_options_en_fr, valid_en_fr, verbose=False)
                valid_err_en_fr = valid_errs_en_fr.mean()
                training_progress_en_fr.history_errs.append(float(valid_err_en_fr))
                if training_progress_en_fr.uidx == 0 or valid_err_en_fr <= numpy.array(training_progress_en_fr.history_errs).min():
                    best_p_en_fr = theano_util.unzip_from_theano(tparams_en_fr, excluding_prefix='prior_')
                    
                valid_errs_fr_en, alignment = nmt.pred_probs(f_log_probs_fr_en, nmt.prepare_data,
                                        model_options_fr_en, valid_fr_en, verbose=False)
                valid_err_fr_en = valid_errs_fr_en.mean()
                training_progress_fr_en.history_errs.append(float(valid_err_fr_en))
                if training_progress_fr_en.uidx == 0 or valid_err_fr_en <= numpy.array(training_progress_fr_en.history_errs).min():
                    best_p_fr_en = theano_util.unzip_from_theano(tparams_fr_en, excluding_prefix='prior_')
                    
                print 'Valid en_fr: ', valid_err_en_fr
                print 'Valid_fr_en: ', valid_err_fr_en
                training_progress_en_fr.save_to_json(training_progress_file_en_fr)
                training_progress_fr_en.save_to_json(training_progress_file_fr_en)
                
                if external_validation_script_en_fr is not None:
                    print "Calling external validation script"
                    if p_validation_en_fr is not None and p_validation_en_fr.poll() is None:
                        print "Waiting for previous validation run to finish"
                        print "If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes"
                        valid_wait_start_en_fr = time.time()
                        p_validation_en_fr.wait()
                        print "Waited for {0:.1f} seconds".format(time.time()-valid_wait_start_en_fr)
                    print 'Saving  model...',
                    params = theano_util.unzip_from_theano(tparams_en_fr)
                    both_params = dict(params)
                    numpy.savez(saveto +'.en_fr.dev', **both_params)
                    json.dump(model_options_en_fr, open('%s.en_fr.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p_validation_en_fr = Popen([external_validation_script_en_fr, args_en_fr])

                if external_validation_script_fr_en is not None:
                    print "Calling external validation script"
                    if p_validation_fr_en is not None and p_validation_fr_en.poll() is None:
                        print "Waiting for previous validation run to finish"
                        print "If this takes too long, consider increasing validation interval, reducing validation set size, or speeding up validation by using multiple processes"
                        valid_wait_start_fr_en = time.time()
                        p_validation_fr_en.wait()
                        print "Waited for {0:.1f} seconds".format(time.time()-valid_wait_start_fr_en)
                    print 'Saving  model...',
                    params = theano_util.unzip_from_theano(tparams_fr_en)
                    both_params = dict(params)
                    numpy.savez(saveto +'.fr_en.dev', **both_params)
                    json.dump(model_options_fr_en, open('%s.fr_en.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p_validation_fr_en = Popen([external_validation_script_fr_en, args_fr_en])
           
             #load for next batch:   
            try:      
                x_en = train_en.next()
                x_en_s, x_mask_en = lm.prepare_data(x_en, maxlen=maxlen,
                                                            n_words=n_words_en)
            except StopIteration:
                break
            
            try: 
                x_fr = train_fr.next()
                x_fr_s, x_mask_fr = lm.prepare_data(x_fr, maxlen=maxlen,
                                                            n_words=n_words_fr)
            except StopIteration:
                break
            
            try:
                x_en_en_fr, x_fr_en_fr = train_en_fr.next()
                x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr, maxlen=maxlen,
                                                            n_words=n_words_en)
            except StopIteration:
                train_en_fr.reset()
                x_en_en_fr, x_fr_en_fr = train_en_fr.next()
                x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr, maxlen=maxlen,
                                                            n_words=n_words_en)
                
            try:
                x_fr_fr_en, x_en_fr_en = train_fr_en.next()
                x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en, maxlen=maxlen,
                                                            n_words=n_words_fr)
            except StopIteration:
                train_fr_en.reset()
                x_fr_fr_en, x_en_fr_en = train_fr_en.next()
                x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en, maxlen=maxlen,
                                                            n_words=n_words_fr)
   
    return 0
