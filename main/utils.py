from scipy.io import loadmat
import numpy as np
import math
from random import shuffle
from random import choice
from random import randint
from theano import tensor as T

def lookupWordsIDX(words,w):
    if w in words:
        return words[w]
    else:
        #print 'find UUUNKKK words',w
        return words['UUUNKKK']

def lookupTaggerIDX(tagger,w):
    w = w.lower()
    if w in tagger:
        return tagger[w]
    else:
        #print 'find UUUNKKK words',w
        return tagger['*']

def lookup_with_unk(We,words,w):
    if w in words:
        return We[words[w],:],False
    else:
        #print 'find Unknown Words in WordSim Task',w
        return We[words['UUUNKKK'],:],True

def lookupwordID(words,array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i] in words):
            result.append(words[array[i]])
        else:
            #print "Find Unknown Words ",w
            result.append(words['UUUNKKK'])
    return result

def lookupTaggerID(tagger, array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i] in tagger):
            result.append(tagger[array[i]])
        else:
            #print "Find Unknown tagger *
            result.append(tagger['*'])
    return result


def getData(f, words, tagger):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
	    index = i.find('|||')
	    if index == -1:
		print('file error\n')
		return None
	    x = i[:index-1]
	    y = i[index+4:-1]
	    x = x.split(' ')
	    y = y.split(' ')
            #print x
	    #print y
	    x = lookupwordID(words, x)
            y = lookupTaggerID(tagger, y)
	    #print y
            X.append(x)
	    Y.append(y)
   
    return X, Y

def getUnlabeledData(f, words):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
            x = i[:-1]
            x = x.split(' ')
            x = lookupwordID(words, x)
            #print y
            X.append(x)
    return X

def getUnlabeledTaggerData(f, words):
    data = open(f,'r')
    lines = data.readlines()
    data.close()
    X = []
    for i in lines:
        if(len(i) > 0):
            x = i[:-1]
            x = x.split(' ')
            X.append(x)
    return X

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n+1
        We.append(v)
    return (words, We)

def getTagger(Taggerfile):
    tag = {}
    f = open(Taggerfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        tag[i] = n
    return tag

def getTaggerList(Taggerfile):
    taglist = []
    f = open(Taggerfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        taglist.append(i)
    return taglist

def getWordlist(textfile):
    words=[]
    words.append('<s>')
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        words.append(i[0])
    return words

def getTaggerData(f, words, tagger):
    data = open(f,'r')
    lines = data.readlines()
    data.close()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
            index = i.find('|||')
            if index == -1:
                print('file error\n')
                re
            x = i[:index-1]
            y = i[index+4:-1]
            x = x.split(' ')
            y = y.split(' ')
            #print x
            #print y
            #x = lookupwordID(words, x)
            y = lookupTaggerID(tagger, y)
            #print y
            X.append(x)
            Y.append(y)

    return X, Y
