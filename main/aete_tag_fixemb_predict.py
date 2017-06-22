import theano
import numpy as np
from theano import tensor as T
from theano import config
import pdb
import random as random
#from evaluate import evaluate_all
import time
import utils
from utils import lookupwordID
import string
from utils import getUnlabeledTaggerData
#from LSTMLayerNoOutput import LSTMLayerNoOutput
from collections import OrderedDict
import lasagne
import sys
import cPickle
import pickle
from utils import getTaggerData


def checkIfQuarter(idx,n):
    #print idx, n
    if idx==round(n/4.) or idx==round(n/2.) or idx==round(3*n/4.):
        return True
    return False

class aetagger_model(object):

    #takes list of seqs, puts them in a matrix
    #returns matrix of seqs and mask	
    def prepare_data(self, list_of_seqs,labels, contextsize, words):
        lengths = [len(s) for s in list_of_seqs]
	sumlength = sum(lengths)
        n_samples = len(list_of_seqs)
        x = np.zeros((sumlength, 2*contextsize+1)).astype('int32')
        y = np.zeros(sumlength).astype('int32')
	D = np.zeros((sumlength, 10)).astype('int32')

	index = 0
        for i in range(n_samples):
	    seq_id = lookupwordID(words, list_of_seqs[i])
	    new_seq = [0]*contextsize + seq_id + [1]*contextsize
            for j in range(lengths[i]):
            	x[index, :] = new_seq[j:j+2*contextsize+1]
            	y[index] = labels[i][j]
		word_j = list_of_seqs[i][j]
                punc_flag = 1
		a =0
                for s in word_j:
                        if s in string.punctuation:
                                a = a +1
                if a == len(word_j):
                        punc_flag = 0

                if word_j =='<@MENTION>':
                        D[index,0]=1
                elif (word_j[0] =='#') and (len(word_j)!= 1):
                        D[index,1]=1
                elif word_j =='rt':
                        D[index, 2]=1

                elif 'URL' in word_j:
                        D[index, 3]=1
                elif word_j.replace('.','',1).isdigit():
                        D[index, 4]=1
                # check whether it is punc
                elif '$' in word_j:
                        D[index, 5]=1
                elif word_j ==':':
                        D[index, 7]=1
                elif word_j =='...':
                        D[index, 8]=1
                elif (len(word_j) == 1) and (word_j[0] in string.punctuation):
                        D[index, 9]=1
		elif punc_flag ==0:
                        D[index,6]=1

		index = index + 1
	#print len(labels)
        return x, y, n_samples, D

    def prepare_aedata(self, list_of_seqs, contextsize, words):
	lengths = [len(s) for s in list_of_seqs]
        sumlength = sum(lengths)
        n_samples = len(list_of_seqs)

	D = np.zeros((sumlength, 10)).astype('int32')
        x = np.zeros((sumlength, 2*contextsize+1)).astype('int32')
        index = 0
        for i in range(n_samples):
	    seq_id = lookupwordID(words, list_of_seqs[i])
            new_seq = [0]*contextsize + seq_id + [1]*contextsize
            for j in range(lengths[i]):
                x[index, :] = new_seq[j:j+2*contextsize+1]
		word_j = list_of_seqs[i][j]
		if len(word_j) ==0:
			print 'error'
			print list_of_seqs[i]
			sys.exit()
                punc_flag = 1
		a = 0
                for s in word_j:
                        if s  in string.punctuation:
                                a = a +1
                if a == len(word_j):
			punc_flag = 0

                if word_j =='<@MENTION>':
                        D[index,0]=1
                elif (word_j[0] =='#') and (len(word_j)!= 1):
                        D[index,1]=1
                elif word_j =='rt':
                        D[index, 2]=1

                elif 'URL' in word_j:
                        D[index, 3]=1
                elif word_j.replace('.','',1).isdigit():
                        D[index, 4]=1
                # check whether it is punc
                elif '$' in word_j:
                        D[index, 5]=1
                elif word_j ==':':
                        D[index, 7]=1
                elif word_j =='...':
                        D[index, 8]=1
                elif (len(word_j) == 1) and (word_j[0] in string.punctuation):
                        D[index, 9]=1
		elif punc_flag ==0:
                        D[index,6]=1

                index = index + 1
        #print len(labels)
        return x, n_samples, D

    def saveParams(self, para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


    
    def __init__(self, We_initial, params):

	self.we = theano.shared(We_initial).astype(theano.config.floatX)
	
	testx = getUnlabeledTaggerData(params.dataf, params.words)
        lengths = [len(s) for s in testx]
        
	contextsize1 = params.window1
        contextsize = params.contextsize
        testx00, test_n_sentense, testd = self.prepare_aedata(testx, contextsize1, params.words)
	testx01, _ , _ = self.prepare_aedata(testx, contextsize, params.words)
	
	g1 = T.imatrix()
	g2 = T.imatrix()
	d = T.imatrix()

	l_in0 = lasagne.layers.InputLayer((None, 2 * contextsize1 + 1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in0, input_size=self.we.eval().shape[0], output_size=self.we.eval().shape[1], W=self.we)
	l_0 = lasagne.layers.ReshapeLayer(l_emb, (-1, (2 * contextsize1 + 1)*params.embedsize))        
       	oldemb0 = lasagne.layers.get_output(l_0, {l_in0:g1})



	l_in1 = lasagne.layers.InputLayer((None, 2 * contextsize + 1))
        l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, input_size=self.we.eval().shape[0], output_size=self.we.eval().shape[1], W=self.we)
       
        l_01 = lasagne.layers.ReshapeLayer(l_emb1, (-1, (2 * contextsize + 1)*params.embedsize))
        l_enc2 = lasagne.layers.DenseLayer(l_01, params.hiddensize)
        l_newemb1 = lasagne.layers.DenseLayer(l_enc2, params.encodesize)	

	# here we use the pretraned model for window size 1, you can choose other models in the model folder	
	f=open('../auencoder_model/e5_w1_l0.05_version1.pickle','r')
        para = pickle.load(f)
        PARA = [p.get_value() for p in para]
        f.close()

        encoder = lasagne.layers.get_all_params(l_newemb1, trainable=True)
	encoder.pop(0)
	for idx, p in enumerate(encoder):
		p.set_value(np.float32(PARA[idx]))
	

	g10 = lasagne.layers.get_output(l_newemb1, {l_in1:g2})
        g1new = T.concatenate((oldemb0, g10, d), axis =1)

	token_function = theano.function([g2], g10)
	
	"""
	tokensize = params.embedsize*(2*params.window1+1) + params.encodesize
	l_in = lasagne.layers.InputLayer((None, tokensize))
	l_1 = lasagne.layers.DenseLayer(l_in, params.taggerhiddensize)
        l_2 = lasagne.layers.DenseLayer(l_1, params.taggerhiddensize)
	l_out = lasagne.layers.DenseLayer(l_2, 25, nonlinearity=lasagne.nonlinearities.softmax)

	output = lasagne.layers.get_output(l_out,{l_in:g1new})
	pred = T.argmax(output, axis =1)

	#y0=T.ivector()
 	#y1 = T.ones_like(y0)

        #SUM = T.sum(y1)
        #acc = 1.0 * T.sum(T.eq(pred, y0))/SUM
	
	pred_function = theano.function([g1, g2, d], pred)
	
        	
		
	c_params0 = lasagne.layers.get_all_params(l_out, trainable=True)
	f=open('../models/tagger_morefeature.pickle','r')
        para = pickle.load(f)
        PARA = [p.get_value() for p in para]
        f.close()
	c_params = c_params0
        for idx, p in enumerate(c_params):
                p.set_value(np.float32(PARA[idx]))
		            
        pred = pred_function(testx00, testx01, testd)
	"""	

	g10_test = token_function(testx01)
	
	# g10_test is the output of the token embedding 
	# testx is the list of tweets. And each element in the list is one tweet.
	data = testx, g10_test
	np.save(params.outfile+ '.npy', data)
	
	"""
	# here is the code for the tagger result
	f = open('TokenEmbedding_tagger_result','w')
        index = 0
        for ii, s in enumerate(lengths):
                for i in range(s):
                        tmp = pred[index]
                        #print tmp
                        f.write(str(i+1) + '\t' + testx[ii][i] +'\t'+  params.taggerlist[testy[ii][i]] + '\t' + params.taggerlist[tmp] + '\n')
                        index = index +1
                f.write('\n')
        f.close()
	"""
