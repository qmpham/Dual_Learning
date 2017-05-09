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
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.))
    
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
        self.tparams['']
        self.cost = cost
        self.x = x
        self.x_mask = x_mask
        self.y = 
        #print "Print out in build_model()"
        #print opt_ret
        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost
    

    
    
    
    
    
    
    
    
    
    
    