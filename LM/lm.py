'''
Build a simple neural language model using GRU units
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl

import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

profile = False
layers = {'ff': ('param_init_fflayer', 'fflayer'),
              'gru': ('param_init_gru', 'gru_layer'),
              }
# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params
    
# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)    

# layers: 'name': ('parameter initializer', 'feedforward')

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
  

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


def gru_layer(tparams, state_below, options, prefix='gru',
              mask=None, one_step=False, init_state=None, **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_,  h_,          U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:  # sampling
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:  # training
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval]
    return rval        
        
        
                
class lm():
    def __init__(self):
        self.tparams = None
        self.params = None
        self.model_options = None
        
    def get_options(self,options):
        self.model_options= options
    
    # initialize Theano shared variables according to the initial parameters
    def init_tparams(self):
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            tparams[kk] = theano.shared(self.params[kk], name=kk)
        self.tparams = tparams
        return tparams
    
    
    # load parameters
    def load_params(self,path):
        pp = numpy.load(path)
        for kk, vv in self.params.iteritems():
            if kk not in pp:
                warnings.warn('%s is not in the archive' % kk)
                continue
            self.params[kk] = pp[kk]
    
        return 0    
    
    # initialize all parameters
    def init_params(self):
        options = self.model_options
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder',
                                                  nin=options['dim_word'],
                                                  dim=options['dim'])
        # readout
        params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                    nin=options['dim'], nout=options['dim_word'],
                                    ortho=False)
        params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                    nin=options['dim_word'],
                                    nout=options['dim_word'], ortho=False)
        params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                    nin=options['dim_word'],
                                    nout=options['n_words'])
        self.params = params
        return params
    
    
    # build a training model
    def build_model(self):
        options = self.model_options
        tparams = self.tparams
        opt_ret = dict()
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.))
    
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')
    
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
    
        # input
        emb = tparams['Wemb'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        opt_ret['emb'] = emb
    
        # pass through gru layer, recurrence here
        proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                                prefix='encoder',
                                                mask=x_mask)
        proj_h = proj[0]
        opt_ret['proj_h'] = proj_h
    
        # compute word probabilities
        logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer('ff')[1](tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev)
        logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit',
                                   activ='linear')
        logit_shp = logit.shape
        probs = tensor.nnet.softmax(
            logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
    
        # cost
        x_flat = x.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx])
        cost = cost.reshape([x.shape[0], x.shape[1]])
        opt_ret['cost_per_sample'] = cost
        cost = (cost * x_mask).sum(0)
    
        return trng, use_noise, x, x_mask, opt_ret, cost

    
