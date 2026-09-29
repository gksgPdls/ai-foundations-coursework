#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    word=x.split() 

    return Counter(word) 

    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights={}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    
    def dotProduct(d1, d2):
        return sum(d1.get(f, 0)*v for f, v in d2.items())
    
    def increment(d1, scale, d2):
        for f, v in d2.items():
            d1[f]=d1.get(f, 0)+scale*v
    
    for i in range(numIters):
        for x, y in trainExamples:
            features=featureExtractor(x)
            prediction=dotProduct(weights, features)
            if prediction*y<=0: 
                increment(weights, eta*y, features)

        trainError=evaluatePredictor(trainExamples, lambda x: 1 if dotProduct(weights, featureExtractor(x))>=0 else -1)
        testError=evaluatePredictor(testExamples, lambda x: 1 if dotProduct(weights, featureExtractor(x))>=0 else -1)
        
        print(f"Iteration {i}: Train Error = {trainError}, Test Error = {testError}")
    
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3d: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        
        x=x.replace(" ", "")
        feature=Counter(x[i:i+n] for i in range(len(x)-n+1))
        
        return feature
        
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    
    centroids=random.sample(examples, K)
    assignments=[-1]*len(examples)  
    
    for iteration in range(maxIters):
        new_assignments=[0]*len(examples)
        
        for i, example in enumerate(examples):
            min_distance=float('inf')
            for j, centroid in enumerate(centroids):
                distance=0
                for f in example.keys():
                    diff=example.get(f, 0)-centroid.get(f, 0)
                    distance+=diff*diff
                    if distance>=min_distance: 
                        break
                if distance<min_distance:
                    min_distance=distance
                    new_assignments[i]=j
        
        new_centroids=[{} for _ in range(K)]
        counts=[0]*K
        
        for i, example in enumerate(examples):
            cluster_id=new_assignments[i]
            counts[cluster_id]+=1
            for f, v in example.items():
                if f in new_centroids[cluster_id]:
                    new_centroids[cluster_id][f]+=v
                else:
                    new_centroids[cluster_id][f]=v
        
        for i in range(K):
            if counts[i]>0:
                for f in new_centroids[i]:
                    new_centroids[i][f]/=counts[i]
        
        if new_assignments==assignments:
            break
        
        centroids=new_centroids
        assignments=new_assignments
    
    loss=0
    for i, example in enumerate(examples):
        centroid=centroids[assignments[i]]
        distance=0
        for f in example.keys():
            diff=example.get(f, 0)-centroid.get(f, 0)
            distance+=diff*diff
        loss+=distance
    
    return centroids, assignments, loss

    # END_YOUR_CODE