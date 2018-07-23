#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:48:56 2018

@author: anil

LDA with ADVI
"""

import numpy as np
import pymc3 as pm, theano.tensor as t

#K = 2 # number of topics
K = 10
V = 12 # number of words
D = 10 # number of documents

#data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
data = np.random.randint(12,size=(30,4))

alpha = np.ones((1, K))
beta = np.ones((1, V))
model = pm.Model()
Wd = [len(doc) for doc in data]
(D, W) = data.shape
n_tokens = np.sum(data[data.nonzero()])
def log_lda(theta,phi):
    def ll_lda(value):  
        dixs, vixs = value.nonzero()
        vfreqs = value[dixs, vixs]
        ll =vfreqs* pm.math.logsumexp(t.log(theta[dixs]) + t.log(phi.T[vixs]), axis = 1).ravel()
        return t.sum(ll) 
    return ll_lda

with model: 
    theta = pm.Dirichlet("thetas", a=alpha, shape=(D, K))
    phi = pm.Dirichlet("phis", a=beta, shape=(K, V))
    doc = pm.DensityDist('doc', log_lda(theta,phi), observed=data)   
with model:    
    inference = pm.ADVI()
    approx = pm.fit(n=1500,method= inference)
tr = approx.sample(draws=1000)
pm.plots.traceplot(tr);    

print(tr)
print(pm.summary(tr))


#import pymc3
#
#with pymc3.Model() as model:
#
#    a = pymc3.Categorical('a', p = [0.25,0.25,0.25,0.25])
#    step = pymc3.ElemwiseCategoricalStep(var=a, values=[0,1,2,3])
#    trace = pymc3.sample(100, step=step)
#
#print(trace['a'])

vix,dix = data.nonzero()
vfeqs = data[vix,dix]

print(theta[vix].eval())