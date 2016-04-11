import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import math
import random
import string
from penne import *

cats_global = ['!','#','$','&',',','@','A','D','E','G','L','M','N','O','P','R','S','T','U','V','X','Y','Z','^','~']

#returns word embeddings as dictionary    
def load_embeddings(filename):
    train_file = open(filename, 'rb')
    ret ={}
    
    for line in train_file:
        line = line.strip().split()
        ret[line[0]] = np.array(line[1:])
        
    return ret
 
#reformats input files as lists of tweets for padding
def words2sentences(lines):
    ret_words = []
    ret_cats = []
    temp_words = []
    temp_cats = []
    lines = lines.readlines()
    total = 0
    for i in xrange(len(lines)):
        line = lines[i].strip().split() 
        if len(line) == 0:
            ret_words.append(temp_words)
            ret_cats.append(temp_cats)
            temp_words = []
            temp_cats = []
            continue
        elif len(line) == 1:
            continue
        else:
            temp_words.append(line[0])
            temp_cats.append(line[1])
            total = total + 1
            
    return ret_words, ret_cats, total
    

#returns list of node indeces to dropout
def dropout(rate):
    num = int(rate * 128)
    inds = random.sample(xrange(0, 128), num)
    return inds
    
    
#makes vectors of extra features to concatenate
def make_features(num, tweet):
    ret = []
    #capitalization
    if num > 0:
        temp = np.zeros(len(tweet))
        for i in xrange(len(tweet)):
            if tweet[i][0].isupper():
                temp[i] = 1
        ret.append(temp)
       
    #punctuation 
    if num > 1:
        temp2 = np.zeros(len(tweet))
        for i in xrange(len(tweet)):
            new = tweet[i].translate(None, string.punctuation)
            if len(new) == 0:
                temp2[i] = 1
        ret.append(temp2)
        
    return ret
        
    
#makes matrix of concatenated word embeddings + extra features 
def make_data_matrix(tweets, w, num_instances, embeddings, extra_feats):
    ret = np.zeros((num_instances, (2*w+1)*50 + (extra_feats*(2*w+1)))) 
    #make pads
    begin = []
    end = []
    for i in xrange(w):
        begin.append("<s>")
        end.append("</s>")  
    
    start = time.time()

    m = 0
    #reads corpus line by line 
    for tweet in tweets:
        line = begin + tweet + end
        line_vecs = [embeddings[x] if x in embeddings.keys() else embeddings['UUUNKKK'] for x in line] 

        for i in xrange(len(tweet)):
            temp = np.concatenate(line_vecs[i:2*w+i+1])
            filler = np.zeros((extra_feats*(2*w+1)),)
            ret[m,:] = np.concatenate((temp, filler))
            
            if extra_feats:
                feats = make_features(extra_feats, line[i:2*w+i+1])
                for feat in feats:
                    ret[m,:] = np.concatenate([ret[m,:(2*w+1)*50]]+feats)
            
            m = m + 1
                    
    end = time.time()    
    return ret
       
       
def train(data, cats, data_test, cats_test, data_final, cats_final, w, extra_feats, reg, drop):
    losses = []
    a_train = []
    a_dev = []
    
    #initialize model parameters
    nh = 128
    #input
    V = parameter(np.random.uniform(-1., 1., (nh, (2*w+1)*50 + (extra_feats*(2*w+1)))))
    a = parameter(np.zeros((nh,)))
    #first layer
    tempW = np.random.uniform(-1., 1., (25, nh))
    if drop:
        inds = dropout(drop)
        for ind in inds:
            tempW[:,ind] = np.zeros((25,))
    W = parameter(tempW)
    b = parameter(np.zeros((25,)))
    
    
    
    i = constant(np.empty(((2*w+1)*50 + (extra_feats*(2*w+1)),)))
    c = constant(np.empty((25,)))
    
    #non-linearities
    h = rectify(dot(V, i) + a)
    o = logsoftmax(rectify(dot(W, h) + b))
    #L2 regularization
    values = compute_values(W)
    r = asum(constant(np.array([(values[W][k][j])**2 for k in xrange(25) for j in xrange(nh)])))
    l = constant(reg)
    #loss function
    e = crossentropy(o, c) + (l*r)
    
    best_acc = 0
    ret = []
    trainer = SGD(learning_rate=0.1)    #SGD = stochastic gradient descents
    for epoch in xrange(50):
        print "EPOCH", epoch
        loss = 0.
        for index in xrange(len(cats)):
            i.value[...] = data[index,:]
            c.value[...] = [1. if x == cats[index] else 0. for x in cats_global]
            loss += trainer.receive(e)      #training model w/ backprop
        print "loss", loss
        losses.append(loss)
        
        #compute accuracies
        train_acc = test(data, cats, w, V, a, W, b, extra_feats)
        a_train.append(train_acc)
        test_acc = test(data_test, cats_test, w, V, a, W, b, extra_feats)
        a_dev.append(test_acc)
        
        #early stopping
        if test_acc > best_acc:
            final_acc = test(data_final, cats_final, w, V, a, W, b, extra_feats)
            print "devtest acc", final_acc, "epoch: ", epoch
            best_acc = test_acc
            ret = [V, a, W, b]
       
    return ret, losses, a_train, a_dev
    
#computes accuracies
def test(data, cats, w, V, a, W, b, extra_feats):
    nh = 128
    
    i = constant(np.empty(((2*w+1)*50 + (extra_feats*(2*w+1)),)))
    c = constant(np.empty((25,)))
    
    h = rectify(dot(V, i) + a)
    o = logsoftmax(rectify(dot(W, h) + b))
    
    correct = 0.
    for index in xrange(len(cats)):
        i.value[...] = data[index,:]
        c.value[...] = [1. if x == cats[index] else 0. for x in cats_global]
        values = compute_values(o)
        j = np.argmax(values[o])
        if cats_global[j] == cats[index]:
            correct += 1
            
    print "acc", (correct/len(cats))
    return (correct/len(cats))
        

def main(argv):
    w = int(sys.argv[1])
    extra_feats = int(sys.argv[2])
    reg = float(sys.argv[3])
    drop = float(sys.argv[4])
    
    #get data
    lines_train = open('tweets-train.txt', 'rb')
    tweets_train = words2sentences(lines_train)
    lines_test = open('tweets-dev.txt', 'rb')
    tweets_test = words2sentences(lines_test)
    lines_final = open('tweets-devtest.txt', 'rb')
    tweets_final = words2sentences(lines_final)
    
    #get embeddings
    embeddings = load_embeddings('embeddings-twitter.txt.gz')
    
    #convert data to vectors
    print "MAKING DATA MATRICES"
    data_train = make_data_matrix(tweets_train[0], w, tweets_train[2], embeddings, extra_feats)
    cats_train = [item for sublist in tweets_train[1] for item in sublist]
    data_test = make_data_matrix(tweets_test[0], w, tweets_test[2], embeddings, extra_feats)
    cats_test = [item for sublist in tweets_test[1] for item in sublist]
    data_final = make_data_matrix(tweets_final[0], w, tweets_final[2], embeddings, extra_feats)
    cats_final = [item for sublist in tweets_final[1] for item in sublist]
    
    #train
    print "TRAINING"
    [V, a, W, b], losses, a_train, a_dev = train(data_train, cats_train, data_test, cats_test, data_final, cats_final, w, extra_feats, reg, drop)
    
    #Make plots
#     plt.plot(range(50), losses, '-', color="red")
#     plt.savefig(str(w)+"w_loss.jpg")
#     plt.close()
#     plt.plot(range(50), a_train, '-', color ="blue")
#     plt.plot(range(50), a_dev, '-', color="green")
#     plt.savefig(str(w)+"w_accuracy.jpg")
    


if __name__ == "__main__":
     main(sys.argv[1:])  