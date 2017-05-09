# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
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

profile = False

from data_iterator import TextIterator
from training_progress import TrainingProgress
import util
import theano_util 
import alignment_util

import layers 
import initializers 
import optimizers
from warnings import warn

class dualnmt():
    def __init__(self):
        self.params = None
        self.tparams = None
        self.model_options = None
        self.worddicts = None
        self.worddicts_r = None
        self.trng = RandomStreams(1234)
        self.f_log_probs = None
        self.f_init = None
        self.f_next = None
        self.use_noise = theano.shared(numpy.float32(0.))
        
        
    def get_options(self,options):
        self.model_options = options
        
    def invert_dict(self):
        if self.model_options['dim_per_factor'] == None:
            if self.model_options['factors'] == 1:
                self.model_options['dim_per_factor'] = [self.model_options['dim_word']]
            else:
                sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
                sys.exit(1)
        # load dictionaries and invert them
        self.worddicts = [None] * len(self.model_options['dictionaries'])
        self.worddicts_r = [None] * len(self.model_options['dictionaries'])
        for ii, dd in enumerate(self.model_options['dictionaries']):
            self.worddicts[ii] = util.load_dict(dd)
            self.worddicts_r[ii] = dict()
            for kk, vv in self.worddicts[ii].iteritems():
                self.worddicts_r[ii][vv] = kk

        if self.model_options['n_words_src'] is None:
            n_words_src = len(self.worddicts[0])
            self.model_options['n_words_src'] = n_words_src
                              
        if self.model_options['n_words'] is None:
            n_words = len(self.worddicts[1])
            self.model_options['n_words'] = n_words
                              
        if self.model_options['tie_encoder_decoder_embeddings']:
            assert (n_words_src == n_words), "When tying encoder and decoder embeddings, source and target vocabulary size must the same"
        if self.worddicts[0] != self.worddicts[1]:
            warn("Encoder-decoder embedding tying is enabled with different source and target dictionaries. This is usually a configuration error")

    def init_params(self):
        options = self.model_options
        params = OrderedDict()
        # embedding
        params = layers.get_layer_param('embedding')(options, params, options['n_words_src'], options['dim_per_factor'], options['factors'], suffix='')
        if not options['tie_encoder_decoder_embeddings']:
            params = layers.get_layer_param('embedding')(options, params, options['n_words'], options['dim_word'], suffix='_dec')
        
        # encoder: bidirectional RNN
        #forward
        params = layers.get_layer_param(options['encoder'])(options, params,
                                                  prefix='encoder',
                                                  nin=options['dim_word'],
                                                  dim=options['dim'])
        #backward
        params = layers.get_layer_param(options['encoder'])(options, params,
                                                  prefix='encoder_r',
                                                  nin=options['dim_word'],
                                                  dim=options['dim'])
        ctxdim = 2 * options['dim']
    
        # init_state, init_cell
        params = layers.get_layer_param('ff')(options, params, prefix='ff_state',
                                    nin=ctxdim, nout=options['dim'])
        # decoder
        params = layers.get_layer_param(options['decoder'])(options, params,
                                                  prefix='decoder',
                                                  nin=options['dim_word'],
                                                  dim=options['dim'],
                                                  dimctx=ctxdim)
        # readout
        params = layers.get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
                                    nin=options['dim'], nout=options['dim_word'],
                                    ortho=False)
        params = layers.get_layer_param('ff')(options, params, prefix='ff_logit_prev',
                                    nin=options['dim_word'],
                                    nout=options['dim_word'], ortho=False)
        params = layers.get_layer_param('ff')(options, params, prefix='ff_logit_ctx',
                                    nin=ctxdim, nout=options['dim_word'],
                                    ortho=False)
    
    
        params = layers.get_layer_param('ff')(options, params, prefix='ff_logit',
                                    nin=options['dim_word'],
                                    nout=options['n_words'],
                                    weight_matrix = not options['tie_decoder_embeddings'])
    
        self.params = params
        
        tparams = theano_util.init_theano_params(params)
        
        self.tparams = tparams
        
        return params
    
    def build_encoder(self, tparams, options, trng, use_noise, x_mask=None, sampling=False):

        x = tensor.tensor3('x', dtype='int64')
        x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')
    
        # for the backward rnn, we just need to invert x
        xr = x[:,::-1]
        if x_mask is None:
            xr_mask = None
        else:
            xr_mask = x_mask[::-1]
    
        n_timesteps = x.shape[1]
        n_samples = x.shape[2]
    
        if options['use_dropout']:
            retain_probability_emb = 1-options['dropout_embedding']
            retain_probability_hidden = 1-options['dropout_hidden']
            retain_probability_source = 1-options['dropout_source']
            if sampling:
                if options['model_version'] < 0.1:
                    rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                    rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
                    emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                    emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
                    source_dropout = theano.shared(numpy.float32(retain_probability_source))
                else:
                    rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                    rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                    emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
                    emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
                    source_dropout = theano.shared(numpy.float32(1.))
            else:
                if options['model_version'] < 0.1:
                    scaled = False
                else:
                    scaled = True
                rec_dropout = layers.shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
                rec_dropout_r = layers.shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
                emb_dropout = layers.shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
                emb_dropout_r = layers.shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
                source_dropout = layers.shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source, scaled)
                source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
        else:
            rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
            emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
    
        # word embedding for forward rnn (source)
        emb = layers.get_layer_constr('embedding')(tparams, x, suffix='', factors= options['factors'])
        if options['use_dropout']:
            emb *= source_dropout
    
        proj = layers.get_layer_constr(options['encoder'])(tparams, emb, options,
                                                prefix='encoder',
                                                mask=x_mask,
                                                emb_dropout=emb_dropout,
                                                rec_dropout=rec_dropout,
                                                truncate_gradient=options['encoder_truncate_gradient'],
                                                profile=profile)
    
    
        # word embedding for backward rnn (source)
        embr = layers.get_layer_constr('embedding')(tparams, xr, suffix='', factors= options['factors'])
        if options['use_dropout']:
            if sampling:
                embr *= source_dropout
            else:
                # we drop out the same words in both directions
                embr *= source_dropout[::-1]
    
        projr = layers.get_layer_constr(options['encoder'])(tparams, embr, options,
                                                 prefix='encoder_r',
                                                 mask=xr_mask,
                                                 emb_dropout=emb_dropout_r,
                                                 rec_dropout=rec_dropout_r,
                                                 truncate_gradient=options['encoder_truncate_gradient'],
                                                 profile=profile)
    
        # context will be the concatenation of forward and backward rnns
        ctx = theano_util.concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        
        #self.ctx = ctx
        
        return x, ctx
    
    def build_model(self):
        tparams = self.tparams
        options = self.model_options
        opt_ret = dict()
    
        trng = self.trng        
        use_noise = self.use_noise
    
        x_mask = tensor.matrix('x_mask', dtype='float32')
        x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
        y = tensor.matrix('y', dtype='int64')
        y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
        y_mask = tensor.matrix('y_mask', dtype='float32')
        y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')
    
        x, ctx = self.build_encoder(tparams, options, trng, use_noise, x_mask, sampling=False)
        n_samples = x.shape[2]
        n_timesteps_trg = y.shape[0]
    
        if options['use_dropout']:
            retain_probability_emb = 1-options['dropout_embedding']
            retain_probability_hidden = 1-options['dropout_hidden']
            retain_probability_target = 1-options['dropout_target']
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            rec_dropout_d = layers.shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            emb_dropout_d = layers.shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            ctx_dropout_d = layers.shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            target_dropout = layers.shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target, scaled)
            target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
        else:
            rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
            emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
            ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))
    
        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
    
        # or you can use the last state of forward + backward encoder rnns
        # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
    
        if options['use_dropout']:
            ctx_mean *= layers.shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
    
        # initial decoder state
        init_state = layers.get_layer_constr('ff')(tparams, ctx_mean, options,
                                        prefix='ff_state', activ='tanh')
    
        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
        emb = layers.get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
        if options['use_dropout']:
            emb *= target_dropout
    
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
    
        # decoder - pass through the decoder conditional gru with attention
        proj = layers.get_layer_constr(options['decoder'])(tparams, emb, options,
                                                prefix='decoder',
                                                mask=y_mask, context=ctx,
                                                context_mask=x_mask,
                                                one_step=False,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout_d,
                                                ctx_dropout=ctx_dropout_d,
                                                rec_dropout=rec_dropout_d,
                                                truncate_gradient=options['decoder_truncate_gradient'],
                                                profile=profile)
        # hidden states of the decoder gru
        proj_h = proj[0]
    
        # weighted averages of context, generated by attention module
        ctxs = proj[1]
    
        if options['use_dropout']:
            proj_h *= layers.shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden, scaled)
            emb *= layers.shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb, scaled)
            ctxs *= layers.shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden, scaled)
    
        # weights (alignment matrix)
        opt_ret['dec_alphas'] = proj[2]
    
        # compute word probabilities
        logit_lstm = layers.get_layer_constr('ff')(tparams, proj_h, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = layers.get_layer_constr('ff')(tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit_ctx = layers.get_layer_constr('ff')(tparams, ctxs, options,
                                       prefix='ff_logit_ctx', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    
        if options['use_dropout']:
            logit *= layers.shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden, scaled)
    
        logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
        logit = layers.get_layer_constr('ff')(tparams, logit, options,
                                prefix='ff_logit', activ='linear', W=logit_W)
    
        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))
    
        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
        cost = -tensor.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)
        self.cost = cost
        self.x = x
        self.x_mask = x_mask
        self.y = y
        self.y_mask = y_mask
        inps = [x, x_mask, y, y_mask]
        self.f_log_probs = theano.function(inps,cost)
        
        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost
    
    def build_sampler(self, return_alignment=False):
        tparams = self.tparams
        options = self.model_options
        trng = self.trng
        use_noise = self.use_noise
        
        if options['use_dropout'] and options['model_version'] < 0.1:
            retain_probability_emb = 1-options['dropout_embedding']
            retain_probability_hidden = 1-options['dropout_hidden']
            retain_probability_source = 1-options['dropout_source']
            retain_probability_target = 1-options['dropout_target']
            rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
            emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
            ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
            target_dropout = theano.shared(numpy.float32(retain_probability_target))
        else:
            rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
            emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
            ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))
    
        x, ctx = self.build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=True)
        n_samples = x.shape[2]
    
        # get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    
        if options['use_dropout'] and options['model_version'] < 0.1:
            ctx_mean *= retain_probability_hidden
    
        init_state = layers.get_layer_constr('ff')(tparams, ctx_mean, options,
                                        prefix='ff_state', activ='tanh')
    
        print >>sys.stderr, 'Building f_init...',
        outs = [init_state, ctx]
        f_init = theano.function([x], outs, name='f_init', profile=profile)
        print >>sys.stderr, 'Done'
    
        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype='int64')
        init_state = tensor.matrix('init_state', dtype='float32')
    
        # if it's the first word, emb should be all zero and it is indicated by -1
        decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
        emb = layers.get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
        if options['use_dropout'] and options['model_version'] < 0.1:
            emb = emb * target_dropout
        emb = tensor.switch(y[:, None] < 0,
                            tensor.zeros((1, options['dim_word'])),
                            emb)
    
    
        # apply one step of conditional gru with attention
        proj = layers.get_layer_constr(options['decoder'])(tparams, emb, options,
                                                prefix='decoder',
                                                mask=None, context=ctx,
                                                one_step=True,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout_d,
                                                ctx_dropout=ctx_dropout_d,
                                                rec_dropout=rec_dropout_d,
                                                truncate_gradient=options['decoder_truncate_gradient'],
                                                profile=profile)
        # get the next hidden state
        next_state = proj[0]
    
        # get the weighted averages of context for this target word y
        ctxs = proj[1]
    
        # alignment matrix (attention model)
        dec_alphas = proj[2]
    
        if options['use_dropout'] and options['model_version'] < 0.1:
            next_state_up = next_state * retain_probability_hidden
            emb *= retain_probability_emb
            ctxs *= retain_probability_hidden
        else:
            next_state_up = next_state
    
        logit_lstm = layers.get_layer_constr('ff')(tparams, next_state_up, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = layers.get_layer_constr('ff')(tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit_ctx = layers.get_layer_constr('ff')(tparams, ctxs, options,
                                       prefix='ff_logit_ctx', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    
        if options['use_dropout'] and options['model_version'] < 0.1:
            logit *= retain_probability_hidden
    
        logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
        logit = layers.get_layer_constr('ff')(tparams, logit, options,
                                prefix='ff_logit', activ='linear', W=logit_W)
    
        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)
    
        # sample from softmax distribution to get the sample
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)
    
        # compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        print >>sys.stderr, 'Building f_next..',
        inps = [y, ctx, init_state]
        outs = [next_probs, next_sample, next_state]
    
        if return_alignment:
            outs.append(dec_alphas)
    
        f_next = theano.function(inps, outs, name='f_next', profile=profile)
        print >>sys.stderr, 'Done'
        self.f_init = f_init
        self.f_next = f_next
        return f_init, f_next
    
    def gen_sample(self, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False,
               return_hyp_graph=False):
        
        f_init = [self.f_init]
        f_next = [self.f_next]
        
        
        # k is the beam size we have
        if k > 1 and argmax:
            assert not stochastic, \
                'Beam search does not support stochastic sampling with argmax'
    
        sample = []
        sample_score = []
        sample_word_probs = []
        alignment = []
        hyp_graph = None
        if stochastic:
            if argmax:
                sample_score = 0
            live_k=k
        else:
            live_k = 1
    
        if return_hyp_graph:
            from hypgraph import HypGraph
            hyp_graph = HypGraph()
    
        dead_k = 0
    
        hyp_samples=[ [] for i in xrange(live_k) ]
        word_probs=[ [] for i in xrange(live_k) ]
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []
        if return_alignment:
            hyp_alignment = [[] for _ in xrange(live_k)]
    
        # for ensemble decoding, we keep track of states and probability distribution
        # for each model in the ensemble
        num_models = len(f_init)
        next_state = [None]*num_models
        ctx0 = [None]*num_models
        next_p = [None]*num_models
        dec_alphas = [None]*num_models
        # get initial state of decoder rnn and encoder context
        for i in xrange(num_models):
            ret = f_init[i](x)
            next_state[i] = numpy.tile( ret[0] , (live_k,1))
            ctx0[i] = ret[1]
        next_w = -1 * numpy.ones((live_k,)).astype('int64')  # bos indicator
    
        # x is a sequence of word ids followed by 0, eos id
        for ii in xrange(maxlen):
            for i in xrange(num_models):
                ctx = numpy.tile(ctx0[i], [live_k, 1])
                inps = [next_w, ctx, next_state[i]]
                ret = f_next[i](*inps)
                # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
                next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
                if return_alignment:
                    dec_alphas[i] = ret[3]
    
                if suppress_unk:
                    next_p[i][:,1] = -numpy.inf
            if stochastic:
                #batches are not supported with argmax: output data structure is different
                if argmax:
                    nw = sum(next_p)[0].argmax()
                    sample.append(nw)
                    sample_score += numpy.log(next_p[0][0, nw])
                    if nw == 0:
                        break
                else:
                    #FIXME: sampling is currently performed according to the last model only
                    nws = next_w_tmp
                    cand_scores = numpy.array(hyp_scores)[:, None] - numpy.log(next_p[-1])
                    probs = next_p[-1]
    
                    for idx,nw in enumerate(nws):
                        hyp_samples[idx].append(nw)
    
    
                    hyp_states=[]
                    for ti in xrange(live_k):
                        hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                        hyp_scores[ti]=cand_scores[ti][nws[ti]]
                        word_probs[ti].append(probs[ti][nws[ti]])
    
                    new_hyp_states=[]
                    new_hyp_samples=[]
                    new_hyp_scores=[]
                    new_word_probs=[]
                    for hyp_sample,hyp_state, hyp_score, hyp_word_prob in zip(hyp_samples,hyp_states,hyp_scores, word_probs):
                        if hyp_sample[-1]  > 0:
                            new_hyp_samples.append(copy.copy(hyp_sample))
                            new_hyp_states.append(copy.copy(hyp_state))
                            new_hyp_scores.append(hyp_score)
                            new_word_probs.append(hyp_word_prob)
                        else:
                            sample.append(copy.copy(hyp_sample))
                            sample_score.append(hyp_score)
                            sample_word_probs.append(hyp_word_prob)
    
                    hyp_samples=new_hyp_samples
                    hyp_states=new_hyp_states
                    hyp_scores=new_hyp_scores
                    word_probs=new_word_probs
    
                    live_k=len(hyp_samples)
                    if live_k < 1:
                        break
    
                    next_w = numpy.array([w[-1] for w in hyp_samples])
                    next_state = [numpy.array(state) for state in zip(*hyp_states)]
            else:
                cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
                probs = sum(next_p)/num_models
                cand_flat = cand_scores.flatten()
                probs_flat = probs.flatten()
                ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]
    
                #averaging the attention weights accross models
                if return_alignment:
                    mean_alignment = sum(dec_alphas)/num_models
    
                voc_size = next_p[0].shape[1]
                # index of each k-best hypothesis
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]
    
                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
                new_word_probs = []
                new_hyp_states = []
                if return_alignment:
                    # holds the history of attention weights for each time step for each of the surviving hypothesis
                    # dimensions (live_k * target_words * source_hidden_units]
                    # at each time step we append the attention weights corresponding to the current target word
                    new_hyp_alignment = [[] for _ in xrange(k-dead_k)]
    
                # ti -> index of k-best hypothesis
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                    if return_alignment:
                        # get history of attention weights for the current hypothesis
                        new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                        # extend the history with current attention weights
                        new_hyp_alignment[idx].append(mean_alignment[ti])
    
    
                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                word_probs = []
                if return_alignment:
                    hyp_alignment = []
    
                # sample and sample_score hold the k-best translations and their scores
                for idx in xrange(len(new_hyp_samples)):
                    if return_hyp_graph:
                        word, history = new_hyp_samples[idx][-1], new_hyp_samples[idx][:-1]
                        score = new_hyp_scores[idx]
                        word_prob = new_word_probs[idx][-1]
                        hyp_graph.add(word, history, word_prob=word_prob, cost=score)
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(copy.copy(new_hyp_samples[idx]))
                        sample_score.append(new_hyp_scores[idx])
                        sample_word_probs.append(new_word_probs[idx])
                        if return_alignment:
                            alignment.append(new_hyp_alignment[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(copy.copy(new_hyp_samples[idx]))
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(copy.copy(new_hyp_states[idx]))
                        word_probs.append(new_word_probs[idx])
                        if return_alignment:
                            hyp_alignment.append(new_hyp_alignment[idx])
                hyp_scores = numpy.array(hyp_scores)
    
                live_k = new_live_k
    
                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
    
                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = [numpy.array(state) for state in zip(*hyp_states)]
    
        # dump every remaining one
        if not argmax and live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])
    
        if not return_alignment:
            alignment = [None for i in range(len(sample))]
    
        return sample, sample_score, sample_word_probs, alignment, hyp_graph

    
        


    
    
    
    
    
    
    
    
    
    
    
