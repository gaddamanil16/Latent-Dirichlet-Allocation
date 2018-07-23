#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:48:56 2018

@author: anil
"""

import numpy as np
import pymc3 as pm, theano.tensor as t

K = 2 # number of topics
V = 2 # number of words
D = 3 # number of documents

data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

alpha = np.ones((1, K))
beta = np.ones((1, V))
model = pm.Model()
Wd = [len(doc) for doc in data]
(D, W) = data.shape
#print(Wd[1])
        
with model: 
    theta = pm.Dirichlet("thetas", a=alpha, shape=(D, K))
    phi = pm.Dirichlet("phis", a=beta, shape=(K, V))
    z = pm.Categorical("zx", p=theta, shape=(W,D))
#    w = pm.Categorical("wx", 
#                       p=t.reshape(phi[z], (D*W, V)), 
#                       observed=data.reshape(D*W))

    w = pm.Categorical("wx", 
                       p=t.reshape(phi[z.T], (D*W, V)), 
                       observed=data.reshape(D*W))
with model:    
#    step1 = pm.Metropolis()
#    tr = pm.sample(1000,step = step1, chains=1)
    tr = pm.sample(1000,chains = 1)
    #tr = pm.ElemwiseCategorical(vars=['zx','wx'], values = [0,1,2])
    pm.plots.traceplot(tr, ['thetas','phis']);    

print(tr)
print(pm.summary(tr))


import pymc3

with pymc3.Model() as model:

    a = pymc3.Categorical('a', p = [0.25,0.25,0.25,0.25])
    step = pymc3.ElemwiseCategoricalStep(var=a, values=[0,1,2,3])
    trace = pymc3.sample(100, step=step)

print(trace['a'])