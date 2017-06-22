import sys
import warnings
from utils import getWordmap
from params import params
from utils import getWordlist
from utils import getTaggerList
from utils import getTagger
import random
import numpy as np
from aete_tag_fixemb_predict import aetagger_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()
      
if __name__ == "__main__":
	
	# Here the file 'sample' incude the tweets
 	params.dataf = '../data/sample'     
        params.hiddensize = 512
        params.window1 = 0
        params.taggerhiddensize = 512
        params.encodesize = 256
	# the context window size
        params.contextsize = 1
        
        (words, We) = getWordmap('../embeddings/wordvects.tw100w5-m40-it2')
	params.words = words

        params.embedsize = len(We[0])
        words.update({'<s>':0})
        a = np.random.rand(len(We[0]))
        newWe = []
        newWe.append(a)
        We = newWe + We
        We = np.asarray(We).astype('float32')
        
	tagger = getTagger('../data/tagger')
        params.tagger = tagger

        taggerlist = getTaggerList('../data/tagger')
	params.taggerlist = taggerlist
        
	wordlist =  getWordlist('../embeddings/wordvects.tw100w5-m40-it2')
        params.wordlist = wordlist
        tm = aetagger_model(We, params)
