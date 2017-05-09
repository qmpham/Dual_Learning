import NMT
import LM
from NMT.dualnmt import dualnmt
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

dataset_en = "data/train10/train10.en.tok"
dataset_fr = "data/train10/train10.fr.tok"
vocal_en = "data/train10/train10.en.tok.pkl"
vocal_fr = "data/train10/train10.fr.tok.pkl"
test_en = "data/test10/test10.en.tok"
test_fr = "data/test10/test10.fr.tok"

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
              saveto='models/model.npz',
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
              use_domain_interpolation=False, # interpolate between an out-domain training corpus and an in-domain training corpus
              domain_interpolation_min=0.1, # minimum (initial) fraction of in-domain training data
              domain_interpolation_max=1.0, # maximum fraction of in-domain training data
              domain_interpolation_inc=0.1, # interpolation increment to be applied each time patience runs out, until maximum amount of interpolation is reached
              domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'], # in-domain parallel training corpus
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
        
    model_options_fr_en["datasets"] = [dataset_fr,dataset_en]
    model_options_fr_en["dictionaries"] = [vocal_fr,vocal_en]
    model_options_en_fr["datasets"] = [dataset_en,dataset_fr]
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
    
    # build models
    
    nmt_en_fr.build_model()
    nmt_fr_en.build_model()
    
    nmt_en_fr.

    return 0

train()