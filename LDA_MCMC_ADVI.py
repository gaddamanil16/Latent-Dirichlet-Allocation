#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:48:56 2018

@author: anil

LDA with ADVI and MCMC
"""

import numpy as np
import pymc3 as pm, theano.tensor as t
import matplotlib.pyplot as plt
K = 4
V = 4 # number of words
D = 10 # number of documents

data = np.random.randint(V,size=(D,4))

alpha = np.ones((1, K))
beta = np.ones((1, V))
model = pm.Model()
model1 = pm.Model()
Wd = [len(doc) for doc in data]
(D, W) = data.shape

def log_lda(theta,phi):
    def ll_lda(value):  
        dixs, vixs = value.nonzero()
        vfreqs = value[dixs, vixs]
        ll =vfreqs* pm.math.logsumexp(t.log(theta[dixs]) + t.log(phi.T[vixs]), axis = 1).ravel()
        return t.sum(ll) 
    return ll_lda

with model1: 
    theta = pm.Dirichlet("theta", a=alpha, shape=(D, K))
    phi = pm.Dirichlet("phi", a=beta, shape=(K, V))
    doc = pm.DensityDist('doc', log_lda(theta,phi), observed=data)   
with model1:    
    inference = pm.ADVI()
    approx = pm.fit(n=10000,method= inference,callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute')])
   
#inference    
tr1 = approx.sample(draws=1000)
pm.plots.traceplot(tr1);    
pm.plot_posterior(tr1, color='LightSeaGreen');

plt.plot(approx.hist)
'''
With MCMC
'''
        
with model: 
    theta = pm.Dirichlet("thetas", a=alpha, shape=(D, K))
    phi = pm.Dirichlet("phis", a=beta, shape=(K, V))
    z = pm.Categorical("zx", p=theta, shape=(W,D))
    w = pm.Categorical("wx", 
                       p=t.reshape(phi[z.T], (D*W, V)), 
                       observed=data.reshape(D*W))
with model:    
    tr = pm.sample(1000,chains = 1)
pm.plots.traceplot(tr, ['thetas','phis']);    

print(tr)
print(pm.summary(tr))
