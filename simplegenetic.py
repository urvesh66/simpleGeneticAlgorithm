#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 01:42:13 2021

@author: patilu
"""

import numpy as np

def mutate_genes(population,prob,bounds):
    
    npop,ngenes = population.shape
    
    
    assert(bounds.shape[0] == ngenes)

    mask = np.less(np.random.rand(ngenes*npop).reshape([npop,ngenes]),prob)
    
    for ipop in range(npop):
        mask = np.less(np.random.rand(ngenes),prob)
        lo = bounds[:,0]
        hi = bounds[:,1]
        randomgenes = lo + np.random.rand(ngenes) * (hi-lo)
        
        population[ipop,mask] = randomgenes[mask]
    
    return population


def copulate(mother,father):

    mask = np.zeros(mother.size,dtype=np.bool)
    mothergenes = np.random.randint(0,high = mother.size-1,size = np.int(np.random.rand()*mother.size))
    mask[mothergenes] = True
    notmask = np.logical_not(mask)
    
    child = np.zeros(mother.size)
    child[mask] = mother[mask]
    child[notmask] = father[notmask]
    
    return child

def next_gen(population,nchildren = None):
    npop,ngenes = population.shape
    if nchildren is None:
        nchildren = npop
    
    newpopulation = np.zeros([nchildren,ngenes])
    for ichild in range(nchildren):
        imother = np.random.randint(0,high=npop-1)
        ifather = np.random.randint(0,high=npop-1)
        child = copulate(population[imother],population[ifather])
        newpopulation[ichild,:] = child
    return newpopulation
    

def fitness(population,function):
    values = np.array([function(genes) for genes in population])
    #error = np.absolute(values - target)
    
    return values

def test_func(x):
    
    return np.power(x-np.array([3,-2,8,0,15,-20,45]),4).sum()

def test_func2(x):
    x1,x2 = x
    return (-1/np.sqrt(x1*x1 + x2*x2)) * (np.sin(x1-x2)*np.sin(x1+x2))**2 
    
#def test_func(XX):
#    x, y = XX
#    return - (x**2) + (2 * x) - (y ** 2) + (4 * y)
#    #return -2 * (w ** 2) + np.sqrt(np.absolute(w)) - (x ** 2) + (6 * x) - (y ** 2) - (2 * y) - (z ** 2) + (4 * z)

def keep_population(error,population):
    
    n25 = int(0.25*error.size)
    accept_index = np.argsort(error)
    accept_index = accept_index[0:n25]
    mask = np.zeros(error.size,dtype = np.bool)
    mask[accept_index] = True

    return population[mask], error[accept_index[0]]
    
    
    
def init_population(npop,ngenes,bounds=None):
    population = np.random.rand(npop*ngenes).reshape([npop,ngenes])
    
    if bounds is None:
        bounds = np.array([[0,1] for i in range(ngenes)])    
        
    if type(bounds) is not np.ndarray:
        bounds = np.array(bounds)
        
    lo = bounds[:,0]
    hi = bounds[:,1]
    population = lo + population*(hi - lo)
    return population, bounds

prob = 0.2
nitermax = 2000
niter = 0

keep_running = True
population,bounds = init_population(1000,7,bounds = [[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100]])
#population,bounds = init_population(1000,2,bounds = [[0,10],[0,10]])
print(population.shape)
while(keep_running):
    
    
    values = fitness(population,test_func)
    fit_population,min_value = keep_population(values,population)
    print(niter,min_value," ".join(map(lambda x: str(x),fit_population[0])))
    if niter < nitermax:
    
        population = next_gen(population,population.shape[0]-fit_population.shape[0])
        population = mutate_genes(population,prob,bounds)
        population = np.append(fit_population,population,axis=0)

        niter += 1
        
    else:
        keep_running = False
        
        fittest = fit_population[0]
        final_error = min_value