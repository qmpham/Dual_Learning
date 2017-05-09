#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:12:38 2017

@author: minhquang
"""

import json
import numpy
a = numpy.float32(0.)
data = open('model.npz.json').read()
model = json.loads(data)
