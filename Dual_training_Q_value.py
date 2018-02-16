#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import LM
from LM import lm
from src import utils
from src.utils import load_data, get_nn_avg_dist, bow_idf, tf_idf, get_idf, read_embeddings, id2id_trans_emb, id2id_emb_idf
#import theano.sandbox.gpuarray
from nematus import nmt,theano_util,data_iterator,util,optimizers, training_progress,util
from nematus.util import *
from nematus.theano_util import *
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

home_dir = "/home/minhquang.pham/"
#assert os.path.isfile(os.path(home_dir))
floatX = theano.config.floatX
numpy_floatX = numpy.typeDict[floatX]

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

def de_factor(seqs):
    new = []
    for seq in seqs:
            ss = []
            for s in seq:
                ss.append(s[0])
            new.append(ss)
    return new    

def dual_second_ascent(sample_target, Q_value, lr1, lr2, alpha, tparams_1, tparams_2, grads_1,\
                       grads_2, inps_1, inps_2, reward, avg_reward, source, target):     
    
    g_shared_1 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_forward_grad_second_shared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    g_shared_2 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_backward_grad_second_shared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    g_up_1 = [(g1, g2) for g1,g2 in zip(g_shared_1,grads_1)]
    g_up_2 = [(g1, -g2) for g1,g2 in zip(g_shared_2,grads_2)]
    
    f_grad_second_shared = theano.function(inps_1 + inps_2 + [reward, sample_target, Q_value], avg_reward, updates = g_up_1 + g_up_2, on_unused_input='ignore')
    
    params_up_1 = [(p , p + lr1 * g) for p,g in zip(theano_util.itemlist(tparams_1), g_shared_1)]
    params_up_2 = [(p , p + lr2 * (1-alpha) * g) for p,g in zip(theano_util.itemlist(tparams_2), g_shared_2)]
    
    f_second_update = theano.function([lr1,lr2], [], updates = params_up_1 + params_up_2, on_unused_input='ignore')
    
    return f_grad_second_shared, f_second_update

def dual_ascent(sample_target, Q_value, lr, tparams, grads, inps, reward, avg_reward, direction):     
    
    g_shared = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_forward_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    g_up = [(g1, g2) for g1,g2 in zip(g_shared,grads)]
        
    f_grad_shared = theano.function(inps + [reward, sample_target, Q_value], avg_reward, updates = g_up, on_unused_input='ignore')
    
    params_up = [(p , p + lr * g) for p,g in zip(theano_util.itemlist(tparams), g_shared)]
    
    f_update = theano.function([lr], [], updates = params_up, on_unused_input='ignore')
    
    return f_grad_shared, f_update

def adadelta_dual_ascent(lr, tparams, grads, inps, reward, avg_reward, direction):
    g_shared = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_forward_grad_shared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    g_squared = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_forward_grad_squared' % (k,direction)) \
                for k,p in tparams.iteritems() ]
    
    x_squared = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_forward_delta_squared' % (k,direction)) \
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
    g_shared_1 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_forward_grad_second_shared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    g_squared_1 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_forward_grad_second_squared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    x_squared_1 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_forward_delta_second_squared' % (k, source, target)) \
                for k,p in tparams_1.iteritems() ]
    
    g_shared_2 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_backward_grad_second_shared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    g_squared_2 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_backward_grad_second_squared' % (k, target, source)) \
                for k,p in tparams_2.iteritems() ]
    x_squared_2 = [ theano.shared(p.get_value()*numpy_floatX(0.),name= '%s_%s_%s_backward_delta_second_squared' % (k, target, source)) \
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

def train(dim_word = 512,  # word vector dimensionality
              dim = 1024,  # the number of LSTM units
              dim_emb = 200,
              factors = 1, # input factors
              dim_per_factor = None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
              encoder = 'gru',
              decoder = 'gru_cond',
	      frozen_rate = 0.8,
	      frozen_Freq = 10000,
	      bi_reduce_rate = 0.5,
              bi_reduce_Freq = 40000,
              lrate_fw = 0.0001,  # learning rate
              lrate_bw = 0.01,
              lrate_bi = 0.001,
              coeff_bi = 0.1,
              print_gradient = True,
              n_words_en = 15000,  # english vocabulary size
              n_words_fr = 15000 ,  # french vocabulary size
              optimizers_ = None,
              optimizers_biling = "sgd",
              maxlen=30,  # maximum length of training sentences
              disp_grad_Freq = 400,
              dispFreq = 100,
              saveFreq = 400,
              print_sample_freq = 5000,
              validFreq = 2000,
              batch_size= 30,
              valid_batch_size = 30,
              save = True,
              warm_start_ = True,
              saveto = home_dir + 'Dual_NMT/models/dual2/model_dual.npz',
              use_dropout = False,
              use_second_update = True,
              using_word_emb = True,
	      sampling = "full_search", # value: "beam_search", "full_search"
              dropout_embedding = 0.2, # dropout for input embeddings (0: no dropout)
              dropout_hidden = 0.2, # dropout for hidden layers (0: no dropout)
              dropout_source = 0, # dropout source words (0: no dropout)
              dropout_target = 0, # dropout target words (0: no dropout)
              reload_ = False,
              tie_encoder_decoder_embeddings = False, # Tie the input embeddings of the encoder and the decoder (first factor only)
              tie_decoder_embeddings = False, # Tie the input embeddings of the decoder with the softmax output embeddings
              encoder_truncate_gradient = -1, # Truncate BPTT gradients in the encoder to this value. Use -1 for no truncation
              decoder_truncate_gradient = -1, # Truncate BPTT gradients in the decoder to this value. Use -1 for no truncation
              alpha_en_fr = 0.005,
              alpha_fr_en = 0.005,
              beta = 10.0,
              knn = 10,
              reward_scale = 1.0,
              clip_c = 1.,
              external_validation_script_en_fr = None,
	      args_en_fr_1 = "",
              args_en_fr_2 = "",
              external_validation_script_fr_en = None,
	      args_fr_en_1 = "",
              args_fr_en_2 = "",
              numb_samplings = 2,              
              length_constraint_scale = 10.0,
              distance = "dist_bag_of_words_center_2_center",
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
              word_emb_en_path = home_dir + "Dual_NMT/data/word_emb/news_emb/vectors-en.txt",
              word_emb_fr_path = home_dir + "Dual_NMT/data/word_emb/news_emb/vectors-fr.txt",
              reload_training_progress = False,
        ):
    
    #ipdb.set_trace()
    print "saveto:",saveto
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
    #stats:
    stats = dict()
    stats["cost_avg_en_fr"] = []
    stats["cost_avg_fr_en"] = []
    stats["cost_ce_avg_en_fr"] = []
    stats["cost_ce_avg_fr_en"] = []

    stats["sim_avg_en_fr"] = []
    stats["forward_avg_en_fr"] = []
    stats["backward_avg_en_fr"] = []
    #stats["length_diff_avg_en_fr"] = []

    stats["sim_avg_fr_en"] = []
    stats["forward_avg_fr_en"] = []
    stats["backward_avg_fr_en"] = []
    #stats["length_diff_avg_fr_en"] = []

    stats["forward_fr_en_rewards"] = []
    stats["forward_en_fr_rewards"] = []
    stats["backward_fr_en_rewards"] = []
    stats["backward_en_fr_rewards"] = []
    save_count_en_fr = 0
    save_count_fr_en = 0

    #hyperparameters:
    alp_en_fr = theano.shared(numpy_floatX(alpha_en_fr),name="alpha_en_fr")
    alp_fr_en = theano.shared(numpy_floatX(alpha_fr_en),name="alpha_fr_en")
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
    # correspondance_matrice_vocab_list_emb
    
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
    probs_en_fr = opt_ret_en_fr["probs"]
    probs_fr_en = opt_ret_fr_en["probs"]
    inps_en_fr = [x_en, x_en_mask, y_fr, y_fr_mask]
    inps_fr_en = [x_fr, x_fr_mask, y_en, y_en_mask]
    print "Done \n"   
    
    #lbuild samplers
    random_pick_en_fr = opt_ret_en_fr["random_pick"]
    random_pick_fr_en = opt_ret_fr_en["random_pick"]
    print "Build samplers ...",
    f_init_en_fr, f_next_en_fr = nmt.build_sampler(tparams_en_fr, model_options_en_fr, use_noise_en_fr, trng_en_fr)
    f_init_fr_en, f_next_fr_en = nmt.build_sampler(tparams_fr_en, model_options_fr_en, use_noise_fr_en, trng_fr_en)
    f_sampler_en_fr = nmt.build_full_sampler(tparams_en_fr, model_options_en_fr, use_noise_en_fr, trng_en_fr)
    f_sampler_fr_en = nmt.build_full_sampler(tparams_fr_en, model_options_fr_en, use_noise_fr_en, trng_fr_en)
    f_sampler_with_prefix_en_fr = nmt.build_full_sampler_from_prefix(tparams_en_fr, model_options_en_fr, use_noise_en_fr, trng_en_fr)
    f_sampler_with_prefix_fr_en = nmt.build_full_sampler_from_prefix(tparams_fr_en, model_options_fr_en, use_noise_fr_en, trng_fr_en)
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

    sample_en_fr = T.matrix("sample_en_fr",dtype="int64") #same size as y_en_fr
    sample_fr_en = T.matrix("sample_fr_fr",dtype="int64") #same size as y_fr_en

    sample_en_fr_flat = sample_en_fr.flatten()
    sample_en_fr_flat_idx = theano.tensor.arange(sample_en_fr_flat.shape[0]) * n_words_fr + sample_en_fr_flat
    log_probs_en_fr_sample = theano.tensor.log(probs_en_fr.flatten()[sample_en_fr_flat_idx])
    log_probs_en_fr_sample = log_probs_en_fr_sample.reshape([sample_en_fr.shape[0], sample_en_fr.shape[1]])
    log_probs_en_fr_sample = log_probs_en_fr_sample * y_fr_mask # sample_en_fr_mask = y_fr_mask
    
    sample_fr_en_flat = sample_fr_en.flatten()
    sample_fr_en_flat_idx = theano.tensor.arange(sample_fr_en_flat.shape[0]) * n_words_en + sample_fr_en_flat
    log_probs_fr_en_sample = theano.tensor.log(probs_fr_en.flatten()[sample_fr_en_flat_idx])
    log_probs_fr_en_sample = log_probs_fr_en_sample.reshape([sample_fr_en.shape[0], sample_fr_en.shape[1]])
    log_probs_fr_en_sample = log_probs_fr_en_sample * y_en_mask # sample_fr_en_mask = y_en_mask


    Q_value_en_fr = T.matrix("Q_value_en_fr")
    Q_value_fr_en = T.matrix("Q_value_fr_en")
        # -cost = log(p(s_mid|s))    
    new_cost_en_fr = T.mean((Q_value_en_fr * log_probs_en_fr_sample).sum(0)/y_fr_mask.sum(0))
    new_cost_fr_en = T.mean((Q_value_fr_en * log_probs_fr_en_sample).sum(0)/y_en_mask.sum(0))
        # cross entropy
    cost_ce_en_fr = cost_en_fr.mean()
    cost_ce_fr_en = cost_fr_en.mean()    

        # gradient newcost = gradient( avg reward * -cost) = avg reward_i * gradient( -cost_i) = avg reward_i * gradient(log p(s_mid | s)) stochastic approximation of policy gradient
    grad_en_fr = T.grad(new_cost_en_fr, wrt = theano_util.itemlist(tparams_en_fr)) 
    grad_fr_en = T.grad(new_cost_fr_en, wrt = theano_util.itemlist(tparams_fr_en)) 
    
    grad_ce_en_fr = T.grad(cost_ce_en_fr, wrt = theano_util.itemlist(tparams_en_fr))
    grad_ce_fr_en = T.grad(cost_ce_fr_en, wrt = theano_util.itemlist(tparams_fr_en))
    clip_c = numpy_floatX(clip_c)
    # apply gradient clipping here
    if clip_c > 0.:
        g1 = numpy_floatX(0.)
        for g in grad_en_fr:
            g1 += (g**2).sum()
        new_grads_1 = []
        for g in grad_en_fr:
            new_grads_1.append(T.switch(g1 > (clip_c**2),
                            g / T.sqrt(g1) * clip_c, g))
        grad_en_fr = new_grads_1
        
        g2 = numpy_floatX(0.)
        for g in grad_fr_en:
            g2 += (g**2).sum()
        new_grads_2 = []
        for g in grad_fr_en:
            new_grads_2.append(T.switch(g2 > (clip_c**2),
                            g / T.sqrt(g2) * clip_c, g))
        grad_fr_en = new_grads_2
        
        g3 =  numpy_floatX(0.)
        for g in grad_ce_en_fr:
            g3 += (g**2).sum()
        new_grads_3 = []
        for g in grad_ce_en_fr:
            new_grads_3.append(T.switch(g3 > (clip_c**2),
                            g / T.sqrt(g3) * clip_c, g))
        grad_ce_en_fr = new_grads_3
        
        g4 =  numpy_floatX(0.)
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
    lr_forward = T.scalar('lrate_forward',dtype=floatX)
    lr_backward = T.scalar('lrate_backward',dtype=floatX)
    lr1 = T.scalar('lrate1',dtype=floatX)
    lr2 = T.scalar('lrate2',dtype=floatX)
    if optimizers_ is not None:
        f_dual_grad_shared_en_fr, f_dual_update_en_fr = eval("%s_dual_ascent"%optimizers_)(sample_en_fr, Q_value_en_fr, lr_forward, tparams_en_fr, grad_en_fr, \
                                                                inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr" ) 
        f_dual_grad_shared_fr_en, f_dual_update_fr_en = eval("%s_dual_ascent"%optimizers_)(sample_fr_en, Q_value_fr_en, lr_forward, tparams_fr_en, grad_fr_en, \
                                                                inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
    else:    
        f_dual_grad_shared_en_fr, f_dual_update_en_fr = dual_ascent(sample_en_fr, Q_value_en_fr, lr_forward, tparams_en_fr, grad_en_fr, \
                                                                    inps_en_fr, reward_en_fr, avg_reward_en_fr, "en_fr" ) 
        f_dual_grad_shared_fr_en, f_dual_update_fr_en = dual_ascent(sample_fr_en, Q_value_fr_en, lr_forward, tparams_fr_en, grad_fr_en, \
                                                                    inps_fr_en, reward_fr_en, avg_reward_fr_en, "fr_en") 
    
    if use_second_update:
        if optimizers_ is not None:
            f_dual_grad_shared_en_fr, f_dual_update_en_fr = eval("%s_second_dual_ascent"%optimizers_)(sample_en_fr, Q_value_en_fr, lr_forward, lr_backward,\
                                                                                         alp_en_fr, tparams_en_fr,\
                                                                                         tparams_fr_en, grad_en_fr,\
                                                                                         grad_ce_fr_en, inps_en_fr, inps_fr_en,\
                                                                                         reward_en_fr, avg_reward_en_fr, "en", "fr")
            f_dual_grad_shared_fr_en, f_dual_update_fr_en = eval("%s_second_dual_ascent"%optimizers_)(sample_fr_en, Q_value_fr_en, lr_forward, lr_backward,\
                                                                                         alp_fr_en, tparams_fr_en,\
                                                                                         tparams_en_fr, grad_fr_en,\
                                                                                         grad_ce_en_fr, inps_fr_en, inps_en_fr,\
                                                                                         reward_fr_en, avg_reward_fr_en, "fr", "en") 
        
        else:
            f_dual_grad_shared_en_fr, f_dual_update_en_fr = dual_second_ascent(sample_en_fr, Q_value_en_fr, lr_forward, lr_backward, alp_en_fr, tparams_en_fr,\
                                                                                             tparams_fr_en, grad_en_fr,\
                                                                                             grad_ce_fr_en, inps_en_fr, inps_fr_en,\
                                                                                             reward_en_fr, avg_reward_en_fr, "en", "fr")
            f_dual_grad_shared_fr_en, f_dual_update_fr_en = dual_second_ascent(sample_fr_en, Q_value_fr_en, lr_forward, lr_backward, alp_fr_en, tparams_fr_en,\
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
    #load word embeddings: 

    #load dictionaries:

    en_word2id = OrderedDict(load_dict(vocab_en))
    fr_word2id = OrderedDict(load_dict(vocab_fr))

    #load word embeddings:
    if using_word_emb:
        print "loading word embedding:"
        en_emb_word2id, en_emb = read_embeddings(word_emb_en_path)
        fr_emb_word2id, fr_emb = read_embeddings(word_emb_fr_path)
        print "done!"
        # load idf index
        print "computing idf index:"
        n_idf = 20000000
        idf, idf_word2id = get_idf({"en":dataset_mono_en,"fr":dataset_mono_fr}, "en", "fr", n_idf)

        en_id2id_1, en_ind_common_1 = id2id_trans_emb(en_word2id, en_emb_word2id)
        en_id2id_2, en_ind_common_2 = id2id_emb_idf(en_emb_word2id, idf_word2id["en"])

        fr_id2id_1, fr_ind_common_1 = id2id_trans_emb(fr_word2id, fr_emb_word2id)
        fr_id2id_2, fr_ind_common_2 = id2id_emb_idf(fr_emb_word2id, idf_word2id["fr"])

        ind_common = {}
        ind_common["en"] = [en_ind_common_1, en_ind_common_2]
        ind_common["fr"] = [fr_ind_common_1, fr_ind_common_2]
        id2id = {}
        id2id["en"] = [en_id2id_1, en_id2id_2]
        id2id["fr"] = [fr_id2id_1, fr_id2id_2]
        emb_dict_size = {}
        emb_dict_size["en"] = en_emb.shape[0]
        emb_dict_size["fr"] = fr_emb.shape[0]
        print "done!"
        
        # define tf_idf vectors:

        en_tf_idf_cbow = T.matrix("src_tf_idf_cbow",dtype="float32")
        fr_tf_idf_cbow = T.matrix("tgt_tf_idf_cbow",dtype="float32")
        #emb based scores

        #emb_score_csls = en_tf_idf_cbow.dot(fr_tf_idf_cbow.transpose()).diagonal() * 2 - en_tf_idf.dot(R_s) - fr_tf_idf.dot(R_t)
        # normalize cbow
        en_tf_idf_cbow = (en_tf_idf_cbow.transpose()/((en_tf_idf_cbow * en_tf_idf_cbow).sum(axis=1))).transpose()
        fr_tf_idf_cbow = (fr_tf_idf_cbow.transpose()/((fr_tf_idf_cbow * fr_tf_idf_cbow).sum(axis=1))).transpose()
        emb_score_cosine = (en_tf_idf_cbow.dot(fr_tf_idf_cbow.transpose())).diagonal()
        #score_csls = theano.function([en_tf_idf, fr_tf_idf], emb_score_csls, profile=profile)
        score_cosine = theano.function([en_tf_idf_cbow, fr_tf_idf_cbow], emb_score_cosine, profile=profile)

    print "Compilation time:", time.time()-u     
    
    #Soft-landing phrase   
    
    max_epochs = 500
    c_fb_batches_en_fr = 0
    c_d_batches_en_fr = 0
    c_fb_batches_fr_en = 0
    c_d_batches_fr_en = 0

    cost_acc_en_fr = []
    cost_ce_acc_en_fr = []
    cost_acc_fr_en = []
    cost_ce_acc_fr_en = []

    sim_acc_en_fr = []
    forward_acc_en_fr = []
    backward_acc_en_fr = []    

    sim_acc_fr_en = []
    forward_acc_fr_en = []
    backward_acc_fr_en = []  

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
        x_en_s, x_mask_en = lm.prepare_data(x_en)
        x_fr = train_fr.next()
        x_fr_s, x_mask_fr = lm.prepare_data(x_fr)
        
        x_en_en_fr, x_fr_en_fr = train_en_fr.next()
        x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr)
        x_fr_fr_en, x_en_fr_en = train_fr_en.next()
        x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en)
        while x_en_s is not None or x_fr_s is not None:
            training_progress_en_fr.uidx += 1
            training_progress_fr_en.uidx += 1
       
        #Dual update
            # play game en->fr:
            u0 = time.time()
            if x_en_s is not None:
                u = time.time()
                c_fb_batches_en_fr += 1
                s_source_en = []
                s_source_en_ = []
                s_mid_fr = []
                s_mid_fr_2 = []
                u1 = time.time()
                for jj in xrange(x_en_s.shape[1]):
                    
                    x_current = x_en_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_en.astype('int64')[:, jj].sum(),:]
                    if len(x_current)>0:
		    #sampling
		        if sampling == "beam_search":
                            stochastic = False
                            sample, score, sample_word_probs, alignment, hyp_graph = nmt.gen_sample([f_init_en_fr],\
                                           [f_next_en_fr],
                                           x_current,
                                           k=12,
                                           maxlen=maxlen,
                                           stochastic=stochastic,
                                           argmax=False,
                                           suppress_unk=False,
                                           return_hyp_graph=False)                                    		                
			    # normalize scores according to sequence lengths
                            normalize = True
		            if normalize:
            		        lengths = numpy.array([len(s) for s in sample])
                                score = score / lengths
                            ind_sample = numpy.argsort(score)[:numb_samplings]
        		    sample = [sample[i] for i in ind_sample]
			    tmp = []
                            tmp_ = []
                            for xs in x_en[jj]:
                                tmp.append([xs])
                                tmp_.append(xs)
                            for ss in sample:
			        if len(ss)>0:
                                    s_mid_fr.append(ss)
                                    s_mid_fr_2.append(ss)
                                    s_source_en.append(tmp)
                                    s_source_en_.append(tmp_)
                        elif sampling == "full_search":
                            sample, _ = f_sampler_en_fr(x_current, numb_samplings , maxlen)	    
                            tmp = []
                            tmp_ = []
                            for xs in x_en[jj]:
                                tmp.append([xs])
                                tmp_.append(xs)
                            sample = [list(numpy.trim_zeros(item)) for item in zip(*sample)]
                            # Build tokens picked by conditional prob given subprefix of each sample:
                            for ss in sample:
			        if len(ss)>0:
                                    s_mid_fr.append(ss)
                                    s_mid_fr_2.append(ss)
                                    s_source_en.append(tmp)
                                    s_source_en_.append(tmp_)
                        if numpy.mod(training_progress_en_fr.uidx,print_sample_freq)==0:
                            print 'source'
                    	    for pos in range(x_en_s.shape[0]):
                        	if x_en_s[pos, jj] == 0:
                            	    break
                                vv = x_en_s[pos, jj]
                        	if vv in worddict_en_r:
                            	    print worddict_en_r[vv],
	                        else:
        	                    print 'UNK',
                	    print
                    	    print 'beam search sampling:'
                    	    for vv in sample:
				for w in vv:
                                    if w == 0:
                                        break
                            	    else:
			    	        if w in worddict_fr_r:
                            	            print worddict_fr_r[w],
                            	        else:
                            	            print 'UNK',
                    	        print

                #print "time sampling one batch:", time.time() - u1
                if len(s_source_en)>0:
                    #sampling:
		    u1 = time.time()
                    s_source_en_tmp = s_source_en                   
                    s_mid_fr_tmp = s_mid_fr
                    s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask = nmt.prepare_data(s_source_en, s_mid_fr)
                    s_mid_fr_2, s_mid_fr_2_mask = lm.prepare_data(s_mid_fr_2)
                    
                    m = numpy.array(random_pick_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask))
                    m = m.reshape((s_mid_fr.shape[0], s_mid_fr.shape[1]))
                    sample = [list(item) for item in zip(*m)]
                    sample = [item + [0] for item in sample]
                    cols_trg_fr = []
                    cols_trg_fr_1 = []
                    cols_src_en = []
                    cols_src_en_ = []
                    init_w = []
                        #create prefix:
                    for i in range(s_mid_fr.shape[1]):
                        for j in range(s_mid_fr_mask.astype('int64')[:,i].sum()):
                            m1 = s_mid_fr[:j,i].tolist()
                            m = s_mid_fr[:j+1,i].tolist()
                            m[j] = sample[i][j]
                            init_w.append(m[-1])
                            cols_trg_fr.append(m)
                            cols_trg_fr_1.append(m1)
                            cols_src_en.append(s_source_en_tmp[i])
                            cols_src_en_.append(s_source_en_[i])
                        # prefix
                    s_source_en_tmp_sample, s_source_en_mask_tmp_sample, s_mid_fr_prefix_1, s_mid_fr_prefix_mask_1 = nmt.prepare_data(cols_src_en, cols_trg_fr_1)
                        #sample from prefix appended sampled word
                    s_source_en_tmp_sample, s_source_en_mask_tmp_sample, s_mid_fr_prefix, s_mid_fr_prefix_mask = nmt.prepare_data(cols_src_en, cols_trg_fr)
                    sample_2, _ = f_sampler_with_prefix_en_fr(s_source_en_tmp_sample, s_source_en_mask_tmp_sample, 1, maxlen, s_mid_fr_prefix, init_w)
                    sample_2 = [list(numpy.trim_zeros(item)) for item in zip(*sample_2)]
                        #concatenate sample + prefix
                    cols_n_trg_fr = []
                    for i in range(len(cols_trg_fr)):
                        cols_n_trg_fr.append(cols_trg_fr[i] + sample_2[i])
                    s_source_en_tmp_sample, s_source_en_mask_tmp_sample, s_mid_fr_sampling, s_mid_fr_sampling_mask = nmt.prepare_data(cols_src_en, cols_n_trg_fr)
                        # Trim s_mid_fr_sampling and s_mid_fr_sampling_mask
                    s_mid_fr_sampling = s_mid_fr_sampling[:min(maxlen+5,s_mid_fr_sampling.shape[0]-1),:]
                    s_mid_fr_sampling_mask = s_mid_fr_sampling_mask[:min(maxlen+5,s_mid_fr_sampling_mask.shape[0]-1),:]
                    s_mid_fr_sampling = numpy.concatenate((s_mid_fr_sampling,numpy.zeros((1,s_mid_fr_sampling.shape[1]))),axis=0).astype("int64")
                    s_mid_fr_sampling_mask = numpy.concatenate((s_mid_fr_sampling_mask,numpy.ones((1,s_mid_fr_sampling_mask.shape[1]))),axis=0).astype(floatX)
                    if numpy.mod(training_progress_en_fr.uidx,print_sample_freq)==0:
                        print 'French sample:'
                        for ss in cols_n_trg_fr:
                            for w in ss:
                                if w == 0:
                                    break
                                else:
                                    if w in worddict_fr_r:
                                        print worddict_fr_r[w],
                                    else:
                                        print 'UNK',
                            print

                    #Calculating Q_value by chunks:
                    forward_en_fr_samplings = []
                    backward_en_fr_sammplings = []
                    sim_CBOW_en_fr_samplings = []
                    forward_en_fr_prefix = []
                    backward_en_fr_prefix = []
                    sim_CBOW_en_fr_prefix = []
                    chunksize = 50
                    for i in xrange(0,s_mid_fr_sampling.shape[1],chunksize): 
                        forward_en_fr_samplings.append(f_log_probs_fr(s_mid_fr_sampling[:,i:i+chunksize], s_mid_fr_sampling_mask[:,i:i+chunksize])/s_mid_fr_sampling_mask[:,i:i+chunksize].sum(0))
                        backward_en_fr_sammplings.append(f_log_probs_fr_en(numpy.reshape(s_mid_fr_sampling[:,i:i+chunksize],(1,s_mid_fr_sampling[:,i:i+chunksize].shape[0],\
                                                         s_mid_fr_sampling[:,i:i+chunksize].shape[1])),s_mid_fr_sampling_mask[:,i:i+chunksize],numpy.reshape(s_source_en_tmp_sample[:,:,i:i+chunksize],\
                                                         (s_source_en_tmp_sample[:,:,i:i+chunksize].shape[1],s_source_en_tmp_sample[:,:,i:i+chunksize].shape[2])),\
                                                         s_source_en_mask_tmp_sample[:,i:i+chunksize])/s_source_en_mask_tmp_sample[:,i:i+chunksize].sum(0))
		        sim_CBOW_en_fr_samplings.append(score_cosine(bow_idf(cols_src_en_[i:i+chunksize], [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb),\
                                                        bow_idf(cols_n_trg_fr[i:i+chunksize], [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb)))
                    
                        forward_en_fr_prefix.append(f_log_probs_fr(s_mid_fr_prefix_1[:,i:i+chunksize], s_mid_fr_prefix_mask_1[:,i:i+chunksize])/s_mid_fr_prefix_mask_1[:,i:i+chunksize].sum(0))
                        backward_en_fr_prefix.append(f_log_probs_fr_en(numpy.reshape(s_mid_fr_prefix_1[:,i:i+chunksize],(1,s_mid_fr_prefix_1[:,i:i+chunksize].shape[0],\
                                                     s_mid_fr_prefix_1[:,i:i+chunksize].shape[1])),s_mid_fr_prefix_mask_1[:,i:i+chunksize],numpy.reshape(s_source_en_tmp_sample[:,:,i:i+chunksize],\
                                                     (s_source_en_tmp_sample[:,:,i:i+chunksize].shape[1],s_source_en_tmp_sample[:,:,i:i+chunksize].shape[2])),\
                                                     s_source_en_mask_tmp_sample[:,i:i+chunksize])/s_source_en_mask_tmp_sample[:,i:i+chunksize].sum(0))
                        sim_CBOW_en_fr_prefix.append(score_cosine(bow_idf(cols_src_en_[i:i+chunksize], [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb),\
                                                     bow_idf(cols_trg_fr_1[i:i+chunksize], [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb)))
                    
                    forward_en_fr_samplings = numpy.concatenate(forward_en_fr_samplings,axis=0)
                    forward_en_fr_prefix = numpy.concatenate(forward_en_fr_prefix,axis=0)
                    backward_en_fr_sammplings = numpy.concatenate(backward_en_fr_sammplings,axis=0)
                    backward_en_fr_prefix = numpy.concatenate(backward_en_fr_prefix,axis=0)
                    sim_CBOW_en_fr_samplings = numpy.concatenate(sim_CBOW_en_fr_samplings,axis=0)
                    sim_CBOW_en_fr_prefix = numpy.concatenate(sim_CBOW_en_fr_prefix,axis=0)

                    forward_en_fr = forward_en_fr_samplings - forward_en_fr_prefix
                    backward_en_fr = backward_en_fr_sammplings - backward_en_fr_prefix
		    sim_CBOW_en_fr = sim_CBOW_en_fr_samplings - sim_CBOW_en_fr_prefix

                    reward_en_fr = forward_en_fr * alpha_en_fr + backward_en_fr * (1-alpha_en_fr) + sim_CBOW_en_fr * beta
                    Q_value_en_fr = numpy.zeros((s_mid_fr.shape[0],s_mid_fr.shape[1])).astype(floatX)
                    sample_en_fr = numpy.zeros((s_mid_fr.shape[0],s_mid_fr.shape[1])).astype("int64")

                    c = 0
                    for i in range(s_mid_fr.shape[1]):
                        for j in range(s_mid_fr_mask.astype("int64")[:,i].sum()):
                            if j==0:
			        Q_value_en_fr[j,i] = forward_en_fr_samplings[c] * alpha_en_fr + backward_en_fr_sammplings[c] * (1-alpha_en_fr) + sim_CBOW_en_fr_samplings[c] * beta
			    else:
                                Q_value_en_fr[j,i] = reward_en_fr[c]
                            sample_en_fr[j,i] = sample[i][j]
                            c = c+1
                    #print "Q_value_en_fr:", Q_value_en_fr
		    #print "forward_en_fr_prefix", forward_en_fr_prefix
                    #print "backward_en_fr_prefix", backward_en_fr_prefix
                    #print "forward_en_fr_samplings",forward_en_fr_samplings
                    #print "backward_en_fr_sammplings",backward_en_fr_sammplings
                    #print "time for Q_value: ", time.time() - u1
                #Time for dual ascent update: average over batch then over samples
                    u1 = time.time()
                    sim_CBOW_en_fr = score_cosine(bow_idf(s_source_en_, [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb), bow_idf(s_mid_fr_tmp, [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb))
                    forward_en_fr = f_log_probs_fr(s_mid_fr_2, s_mid_fr_2_mask)/s_mid_fr_2_mask.sum(0)
                    backward_en_fr = f_log_probs_fr_en(numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                 s_mid_fr_mask, \
                                                 numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                 s_source_en_mask)/s_source_en_mask.sum(0)
                    if numpy.mod(save_count_en_fr,3000) == 0:
                        stats["forward_en_fr_rewards"].append(forward_en_fr)
                        stats["backward_en_fr_rewards"].append(backward_en_fr)
                        save_count_en_fr = save_count_en_fr + 1

                    try:
                        reward_en_fr = forward_en_fr * alpha_en_fr + backward_en_fr * (1-alpha_en_fr) + sim_CBOW_en_fr * beta
                    except ValueError:
                        ipdb.set_trace()
                    #print "time to calculate reward: ", time.time()-u1
                    u1 = time.time()
                    if use_second_update:
                        cost_en_fr = f_dual_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask,\
                                                          numpy.reshape(s_mid_fr,(1,s_mid_fr.shape[0],s_mid_fr.shape[1])), \
                                                          s_mid_fr_mask, \
                                                          numpy.reshape(s_source_en,(s_source_en.shape[1],s_source_en.shape[2])),\
                                                          s_source_en_mask, reward_en_fr, sample_en_fr, Q_value_en_fr)
                        f_dual_update_en_fr(lrate_fw,lrate_bw)
                    else:
                        cost_en_fr = f_dual_grad_shared_en_fr(s_source_en, s_source_en_mask, s_mid_fr, s_mid_fr_mask, reward_en_fr, sample_en_fr, Q_value_en_fr)
                        f_dual_update_en_fr(lrate_fw)

                    cost_acc_en_fr.append(cost_en_fr)
                    sim_acc_en_fr.append(numpy.mean(sim_CBOW_en_fr))
                    forward_acc_en_fr.append(numpy.mean(forward_en_fr))
                    backward_acc_en_fr.append(numpy.mean(backward_en_fr))
                    #length_diff_acc_en_fr += numpy.mean(length_constraint_en_fr)

                    if numpy.isnan(cost_en_fr):
                        ipdb.set_trace()
                    #print "time for updatee:", time.time() - u1
                    
                    #print "time for a batch:", time.time() - u
                    
            #play fr --> en:
            if x_fr_s is not None:
                c_fb_batches_fr_en += 1
                s_source_fr = []
                s_source_fr_ = []
                s_mid_en = []
                s_mid_en_2 = []
                u1 = time.time()
                u = time.time()
                for jj in xrange(x_fr_s.shape[1]):                    
                    x_current = x_fr_s[:, jj][None, :, None]
                    # remove padding
                    x_current = x_current[:,:x_mask_fr.astype('int64')[:, jj].sum(),:]
		    if len(x_current)>0:
                    #sampling
		        if sampling == "beam_search":
                            stochastic = False
                            sample, score, sample_word_probs, alignment, hyp_graph = nmt.gen_sample([f_init_fr_en],\
                                           [f_next_fr_en],
                                           x_current,
                                           k = 12,
                                           maxlen = maxlen,
                                           stochastic = stochastic,
                                           argmax = False,
                                           suppress_unk = False,
                                           return_hyp_graph = False)
			    # normalize scores according to sequence lengths
                            normalize = True
                            if normalize:
                                lengths = numpy.array([len(s) for s in sample])
                                score = score / lengths
                            ind_sample = numpy.argsort(score)[:numb_samplings]
                            sample = [sample[i] for i in ind_sample]

                            tmp = []
                            tmp_ = []
                            for xs in x_fr[jj]:
                                tmp.append([xs])
                                tmp_.append(xs)
                            for ss in sample:
                                s_mid_en.append(ss)
                                s_mid_en_2.append(ss)
                                s_source_fr.append(tmp)
                                s_source_fr_.append(tmp_)
                        elif sampling == "full_search":
		            sample, _ = f_sampler_fr_en(x_current, numb_samplings , maxlen)
                            tmp = []
                            tmp_ = []
                            for xs in x_fr[jj]:
                                tmp.append([xs])
                                tmp_.append(xs)
                            for ss in sample:
			        if len(ss)>0:
                                    s_mid_en.append(ss)
                                    s_mid_en_2.append(ss)
                                    s_source_fr.append(tmp)                        
                                    s_source_fr_.append(tmp_)
                        #print sample
                        if numpy.mod(training_progress_en_fr.uidx,print_sample_freq)==0:
                            print 'source'
                            for pos in range(x_fr_s.shape[0]):
                                if x_fr_s[pos, jj] == 0:
                                    break
                                vv = x_fr_s[pos, jj]
                                if vv in worddict_fr_r:
                                    print worddict_fr_r[vv],
                                else:
                                    print 'UNK',
                            print
                            print 'beam search sampling'
                            for vv in sample:
				for w in vv:
                                    if w == 0:
                                    	break
                                    else:
                                    	if w in worddict_en_r:
                                            print worddict_en_r[w],
                                    	else:
                                            print 'UNK',
                            	print

		if len(s_source_fr)>0:
		    #sampling:
                    u1 = time.time()
                    u = time.time()
                    s_source_fr_tmp = s_source_fr                    
                    s_mid_en_tmp = s_mid_en
                    s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask = nmt.prepare_data(s_source_fr, s_mid_en)
                    s_mid_en_2, s_mid_en_2_mask = lm.prepare_data(s_mid_en_2)

                    m = numpy.array(random_pick_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask))
                    m = m.reshape((s_mid_en.shape[0], s_mid_en.shape[1]))
                    sample = [list(item) for item in zip(*m)]
                    sample = [item + [0] for item in sample]
                    cols_trg_en = []
                    cols_trg_en_1 = []
                    cols_src_fr = []
                    cols_src_fr_ = []
                    init_w = []
                        #create prefix:
                    for i in range(s_mid_en.shape[1]):
                        for j in range(s_mid_en_mask.astype('int64')[:,i].sum()):
                            m1 = s_mid_en[:j,i].tolist()
                            m = s_mid_en[:j+1,i].tolist()
                            m[j] = sample[i][j]
                            init_w.append(m[-1])
                            cols_trg_en.append(m)
                            cols_trg_en_1.append(m1)
                            cols_src_fr.append(s_source_fr_tmp[i])
                            cols_src_fr_.append(s_source_fr_[i])
                        # prefix
                    s_source_fr_tmp_sample, s_source_fr_mask_tmp_sample, s_mid_en_prefix_1, s_mid_en_prefix_mask_1 = nmt.prepare_data(cols_src_fr, cols_trg_en_1)
                        #sample from prefix appended sampled word
                    s_source_fr_tmp_sample, s_source_fr_mask_tmp_sample, s_mid_en_prefix, s_mid_en_prefix_mask = nmt.prepare_data(cols_src_fr, cols_trg_en)
                    sample_2, _ = f_sampler_with_prefix_fr_en(s_source_fr_tmp_sample, s_source_fr_mask_tmp_sample, 1, maxlen, s_mid_en_prefix, init_w)
                    sample_2 = [list(numpy.trim_zeros(item)) for item in zip(*sample_2)]
                        #concatenate sample + prefix
                    cols_n_trg_en = []
                    for i in range(len(cols_trg_en)):      
                        cols_n_trg_en.append(cols_trg_en[i] + sample_2[i])
                    s_source_fr_tmp_sample, s_source_fr_mask_tmp_sample, s_mid_en_sampling, s_mid_en_sampling_mask = nmt.prepare_data(cols_src_fr, cols_n_trg_en)
		        # Trim s_mid_en_sampling and s_mid_en_sampling_mask
                    s_mid_en_sampling = s_mid_en_sampling[:min(maxlen+5,s_mid_en_sampling.shape[0]-1),:]
                    s_mid_en_sampling_mask = s_mid_en_sampling_mask[:min(maxlen+5,s_mid_en_sampling_mask.shape[0]-1),:]
                    s_mid_en_sampling = numpy.concatenate((s_mid_en_sampling,numpy.zeros((1,s_mid_en_sampling.shape[1]))),axis=0).astype("int64")
                    s_mid_en_sampling_mask = numpy.concatenate((s_mid_en_sampling_mask,numpy.ones((1,s_mid_en_sampling_mask.shape[1]))),axis=0).astype(floatX)
                    if numpy.mod(training_progress_en_fr.uidx,print_sample_freq)==0:
                        print 'English sample:'
                        for ss in cols_n_trg_en:
                            for w in ss:
                                if w == 0:
                                    break
                                else:
                                    if w in worddict_en_r:
                                        print worddict_en_r[w],
                                    else:
                                        print 'UNK',
                            print

                    #Calculating Q_value by chunks
                    #print "time sampling prefix:", time.time() - u1
                    u1 = time.time()
                    forward_fr_en_samplings = []
                    backward_fr_en_sammplings = []
                    sim_CBOW_fr_en_samplings = []
                    forward_fr_en_prefix = []
                    backward_fr_en_prefix = []
                    sim_CBOW_fr_en_prefix = []
                    chunksize = 50

                    for i in xrange(0, s_mid_en_sampling.shape[1],chunksize):
                        forward_fr_en_samplings.append(f_log_probs_en(s_mid_en_sampling[:,i:i+chunksize], s_mid_en_sampling_mask[:,i:i+chunksize])/s_mid_en_sampling_mask[:,i:i+chunksize].sum(0))
                        backward_fr_en_sammplings.append(f_log_probs_en_fr(numpy.reshape(s_mid_en_sampling[:,i:i+chunksize],(1,s_mid_en_sampling[:,i:i+chunksize].shape[0],\
                                                         s_mid_en_sampling[:,i:i+chunksize].shape[1])),s_mid_en_sampling_mask[:,i:i+chunksize],numpy.reshape(s_source_fr_tmp_sample[:,:,i:i+chunksize],\
                                                         (s_source_fr_tmp_sample[:,:,i:i+chunksize].shape[1],s_source_fr_tmp_sample[:,:,i:i+chunksize].shape[2])),\
                                                         s_source_fr_mask_tmp_sample[:,i:i+chunksize])/s_source_fr_mask_tmp_sample[:,i:i+chunksize].sum(0))
                    #print "time compute backward_forward reward", time.time() - u1
                    #u1 = time.time()
                        sim_CBOW_fr_en_samplings.append(score_cosine(bow_idf(cols_n_trg_en[i:i+chunksize], [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb),\
                                                                bow_idf(cols_src_fr_[i:i+chunksize], [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb)))
                    #print "time compute dist CBOW:", time.time() - u1
                    #u1 = time.time()
                        forward_fr_en_prefix.append(f_log_probs_en(s_mid_en_prefix_1[:,i:i+chunksize], s_mid_en_prefix_mask_1[:,i:i+chunksize])/s_mid_en_prefix_mask_1[:,i:i+chunksize].sum(0))
                        backward_fr_en_prefix.append(f_log_probs_en_fr(numpy.reshape(s_mid_en_prefix_1[:,i:i+chunksize],(1,s_mid_en_prefix_1[:,i:i+chunksize].shape[0],\
                                                     s_mid_en_prefix_1[:,i:i+chunksize].shape[1])),s_mid_en_prefix_mask_1[:,i:i+chunksize],numpy.reshape(s_source_fr_tmp_sample[:,:,i:i+chunksize],\
                                                     (s_source_fr_tmp_sample[:,:,i:i+chunksize].shape[1],s_source_fr_tmp_sample[:,:,i:i+chunksize].shape[2])),\
                                                     s_source_fr_mask_tmp_sample[:,i:i+chunksize])/s_source_fr_mask_tmp_sample[:,i:i+chunksize].sum(0))
                    #print "time compute backward_forward reward", time.time() - u1
                    #u1 = time.time()
                        sim_CBOW_fr_en_prefix.append(score_cosine(bow_idf(cols_trg_en_1[i:i+chunksize], [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb),\
                                                             bow_idf(cols_src_fr_[i:i+chunksize], [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb)))
                    #print "time compute dist CBOW:", time.time() - u1
                    #u1 = time.time()
                    forward_fr_en_samplings = numpy.concatenate(forward_fr_en_samplings,axis=0)
                    forward_fr_en_prefix = numpy.concatenate(forward_fr_en_prefix,axis=0)
                    backward_fr_en_sammplings = numpy.concatenate(backward_fr_en_sammplings,axis=0)
                    backward_fr_en_prefix = numpy.concatenate(backward_fr_en_prefix,axis=0)
                    sim_CBOW_fr_en_samplings = numpy.concatenate(sim_CBOW_fr_en_samplings,axis=0)
                    sim_CBOW_fr_en_prefix = numpy.concatenate(sim_CBOW_fr_en_prefix,axis=0)
                              
		    forward_fr_en = forward_fr_en_samplings - forward_fr_en_prefix
                    backward_fr_en = backward_fr_en_sammplings - backward_fr_en_prefix
                    sim_CBOW_fr_en = sim_CBOW_fr_en_samplings - sim_CBOW_fr_en_prefix
                    reward_fr_en = forward_fr_en * alpha_fr_en + backward_fr_en * (1-alpha_fr_en) + sim_CBOW_fr_en * beta
                    Q_value_fr_en = numpy.zeros((s_mid_en.shape[0],s_mid_en.shape[1])).astype(floatX)
                    sample_fr_en = numpy.zeros((s_mid_en.shape[0],s_mid_en.shape[1])).astype("int64")
                    c = 0
                    for i in range(s_mid_en.shape[1]):
                        for j in range(s_mid_en_mask.astype("int64")[:,i].sum()):
			    if j==0:
				Q_value_fr_en[j,i] = forward_fr_en_samplings[c] * alpha_fr_en + backward_fr_en_sammplings[c] * (1-alpha_fr_en) + sim_CBOW_fr_en_samplings[c] * beta
			    else:
                                Q_value_fr_en[j,i] = reward_fr_en[c]
                            sample_fr_en[j,i] = sample[i][j]
                            c = c+1
                    #print "Q_value_fr_en:", Q_value_fr_en
                    #print "time Q_value:", time.time() - u1
                    u1 = time.time()  
                    
                    sim_CBOW_fr_en = score_cosine(bow_idf(s_mid_en_tmp, [en_ind_common_1, en_ind_common_2], [en_id2id_1, en_id2id_2], idf["en"], en_emb), bow_idf(s_source_fr_, [fr_ind_common_1, fr_ind_common_2], [fr_id2id_1, fr_id2id_2], idf["fr"], fr_emb))
                    
                    forward_fr_en = f_log_probs_en(s_mid_en_2, s_mid_en_2_mask)/s_mid_en_2_mask.sum(0)
                    backward_fr_en = f_log_probs_en_fr(numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                 s_mid_en_mask, \
                                                 numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                 s_source_fr_mask)/s_source_fr_mask.sum(0)                
                    if numpy.mod(save_count_fr_en,3000) == 0:
             	        stats["forward_fr_en_rewards"].append(forward_fr_en)
                        stats["backward_fr_en_rewards"].append(backward_fr_en)
                        save_count_fr_en = save_count_fr_en + 1                              
                #Time for dual ascent update: average over batch then over samples
                    try:
                        reward_fr_en = forward_fr_en * alpha_en_fr + backward_fr_en * (1-alpha_en_fr) + sim_CBOW_fr_en * beta
                    except ValueError:
                        ipdb.set_trace()
                    #print "time to calculate reward: ", time.time() - u1                                                                        
                    u1 = time.time()
                    if use_second_update:
                        cost_fr_en = f_dual_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask,\
                                                          numpy.reshape(s_mid_en,(1,s_mid_en.shape[0],s_mid_en.shape[1])), \
                                                          s_mid_en_mask, \
                                                          numpy.reshape(s_source_fr,(s_source_fr.shape[1],s_source_fr.shape[2])),\
                                                          s_source_fr_mask, reward_fr_en, sample_fr_en, Q_value_fr_en)
                        f_dual_update_fr_en(lrate_fw,lrate_bw)
                    else:
                        cost_fr_en = f_dual_grad_shared_fr_en(s_source_fr, s_source_fr_mask, s_mid_en, s_mid_en_mask, reward_fr_en, sample_fr_en, Q_value_fr_en)
                        f_dual_update_fr_en(lrate_fw)

                    cost_acc_fr_en.append(cost_fr_en)
                    sim_acc_fr_en.append(numpy.mean(sim_CBOW_fr_en))
                    forward_acc_fr_en.append(numpy.mean(forward_fr_en))
                    backward_acc_fr_en.append(numpy.mean(backward_fr_en))
                    #length_diff_acc_fr_en += numpy.mean(length_constraint_fr_en)
                    #print "time to dual update :", time.time()-u1                    
                    if numpy.isnan(cost_fr_en):
                        ipdb.set_trace()
                    #print "time for a batch:", time.time()-u

                        
        #Standard-using bilingual setence pair update 
            #update en->fr model's parameters
            
            u1 = time.time()
            if x_en_en_fr is not None:
                c_d_batches_en_fr += 1
                cost_ce_en_fr = f_grad_shared_en_fr(x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr)
                cost_ce_acc_en_fr += cost_ce_en_fr
                # do the update on parameters
                f_update_en_fr(lrate_bi * coeff_bi)
            
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
                f_update_fr_en(lrate_bi * coeff_bi)
            #print "time for a batch:", time.time()-u0
	    #u0 = time.time()
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
                cost_avg_en_fr = numpy.mean(cost_acc_en_fr) #/ float(c_fb_batches_en_fr)
                cost_avg_fr_en = numpy.mean(cost_acc_fr_en) #/ float(c_fb_batches_fr_en)

                cost_ce_avg_en_fr = numpy.mean(cost_ce_acc_en_fr) #/ float(c_d_batches_en_fr)
                cost_ce_avg_fr_en = numpy.mean(cost_ce_acc_fr_en) #/ float(c_d_batches_fr_en)

                sim_avg_en_fr = numpy.mean(sim_acc_en_fr) #/ float(c_fb_batches_en_fr)
                forward_avg_en_fr = numpy.mean(forward_acc_en_fr) #/ float(c_fb_batches_en_fr)
                backward_avg_en_fr = numpy.mean(backward_acc_en_fr) #/ float(c_fb_batches_en_fr)
                #length_diff_avg_en_fr = length_diff_acc_en_fr / float(c_fb_batches_en_fr)
                
                sim_avg_fr_en = numpy.mean(sim_acc_fr_en) #/ float(c_fb_batches_fr_en)
                forward_avg_fr_en = numpy.mean(forward_acc_fr_en) #/ float(c_fb_batches_fr_en)
                backward_avg_fr_en = numpy.mean(backward_acc_fr_en) #/ float(c_fb_batches_fr_en)
                #length_diff_avg_fr_en = length_diff_acc_fr_en / float(c_fb_batches_fr_en)
                
                #save stats
                stats["cost_avg_en_fr"].append(cost_avg_en_fr)
	        stats["cost_avg_fr_en"].append(cost_avg_fr_en)
    	        stats["cost_ce_avg_en_fr"].append(cost_ce_avg_en_fr)
    		stats["cost_ce_avg_fr_en"].append(cost_ce_avg_fr_en)
    
    		stats["sim_avg_en_fr"].append(sim_avg_en_fr)
    		stats["forward_avg_en_fr"].append(forward_avg_en_fr)
    		stats["backward_avg_en_fr"].append(backward_avg_en_fr)
    		#stats["length_diff_avg_en_fr"].append(length_diff_avg_en_fr)

    		stats["sim_avg_fr_en"].append(sim_avg_fr_en)
    		stats["forward_avg_fr_en"].append(forward_avg_fr_en)
    		stats["backward_avg_fr_en"].append(backward_avg_fr_en)
    		#stats["length_diff_avg_fr_en"].append(length_diff_avg_fr_en)


                print 'epoch:', training_progress_en_fr.eidx , 'Update: ', training_progress_en_fr.uidx,\
                "cost_en_fr: %f cost_fr_en: %f" % (cost_avg_en_fr, cost_avg_fr_en),\
                "cost_ce_en_fr: %f cost_ce_fr_en: %f" % (cost_ce_avg_en_fr, cost_ce_avg_fr_en),\
                "forward_avg_en_fr: %f forward_avg_fr_en: %f" % (forward_avg_en_fr, forward_avg_fr_en),\
		"backward_avg_en_fr: %f backward_avg_fr_en: %f" % (backward_avg_en_fr, backward_avg_fr_en),\
                'UD: ', ud,\
                "sim_avg_en_fr: %f sim_avg_fr_en: %f" % (sim_avg_en_fr, sim_avg_fr_en)
                #print "forward_avg_en_fr: %f forward_avg_fr_en: %f" % (forward_avg_en_fr, forward_avg_fr_en),\
                #"backward_avg_en_fr: %f backward_avg_fr_en: %f" % (backward_avg_en_fr, backward_avg_fr_en),\
		#'UD: ', ud
                #"length_diff_avg_en_fr: %f length_diff_avg_fr_en: %f" % (length_diff_avg_en_fr, length_diff_avg_fr_en),\
                #'UD: ', ud
                
               	c_fb_batches_en_fr = 0
	        c_d_batches_en_fr = 0
    		c_fb_batches_fr_en = 0
    		c_d_batches_fr_en = 0

    		cost_acc_en_fr = 0
    		cost_ce_acc_en_fr = 0
    		cost_acc_fr_en = 0
    		cost_ce_acc_fr_en = 0

    		#dist_acc_en_fr = 0
    		forward_acc_en_fr = 0
    		backward_acc_en_fr = 0
    		#length_diff_acc_en_fr = 0

    		#dist_acc_fr_en = 0
    		forward_acc_fr_en = 0
    		backward_acc_fr_en = 0
    		#length_diff_acc_fr_en = 0

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
                    p_validation_en_fr = Popen([external_validation_script_en_fr, args_en_fr_1, args_en_fr_2, str(training_progress_en_fr.uidx)])

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
                    p_validation_fr_en = Popen([external_validation_script_fr_en, args_fr_en_1, args_fr_en_2, str(training_progress_fr_en.uidx)])

                if frozen_rate > 0.0:
                    if numpy.mod(training_progress_en_fr.uidx, frozen_Freq)==0:
                        lrate_fw = lrate_fw * frozen_rate  # learning rate
                        lrate_bw = lrate_bw * frozen_rate
                        lrate_bi = lrate_bi * frozen_rate
                if bi_reduce_rate > 0.0:
                    if numpy.mod(training_progress_en_fr.uidx, bi_reduce_Freq)==0:
                        lrate_bi = lrate_bi * bi_reduce_rate
            
             #load for next batch:   
            try:      
                x_en = train_en.next()
                x_en_s, x_mask_en = lm.prepare_data(x_en)
            except StopIteration:
                break
            
            try: 
                x_fr = train_fr.next()
                x_fr_s, x_mask_fr = lm.prepare_data(x_fr)
            except StopIteration:
                break
            
            try:
                x_en_en_fr, x_fr_en_fr = train_en_fr.next()
                x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr)
            except StopIteration:
                train_en_fr.reset()
                x_en_en_fr, x_fr_en_fr = train_en_fr.next()
                x_en_en_fr, x_mask_en_en_fr, x_fr_en_fr, x_mask_fr_en_fr = nmt.prepare_data(x_en_en_fr, x_fr_en_fr)
                
            try:
                x_fr_fr_en, x_en_fr_en = train_fr_en.next()
                x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en)
            except StopIteration:
                train_fr_en.reset()
                x_fr_fr_en, x_en_fr_en = train_fr_en.next()
                x_fr_fr_en, x_mask_fr_fr_en, x_en_fr_en, x_mask_en_fr_en = nmt.prepare_data(x_fr_fr_en, x_en_fr_en)
            
    return 0

