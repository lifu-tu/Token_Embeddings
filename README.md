# Token_Embedding

repl4NLP2017

Code for token embedding from "Learning to Embed Words in Context for  Syntactic Tasks"
The code is written in python and requires numpy, theano, and the lasagne libraries.

##Directory Structure

Token_Embedding/embeddings          ---      pretrained 100-dimensional skip-gram embeddings on
                                                   56 million English tweets using the word2vec
Token_Embedding/auencoder_model     ---      pretrained token embedding models
Token_Embedding/main/               ---      code to output token embedding model for different context
                                                                                window


##Compiling

run the following command
```
> ./install.sh
```
This will download the pretrained tweet word embeddings on 56 million English tweets using the word2vec




If you use the code for your work please cite:

@inproceedings{tu-17,
  title={Learning to Embed Words in Context for Syntactic Tasks},
  author={Lifu Tu and Kevin Gimpel and Karen Livescu},
  booktitle={Proc. of RepL4NLP},
  year={2017}
}


@inproceedings{tu-17-long,
  title={Learning to Embed Words in Context for Syntactic Tasks},
  author={Lifu Tu and Kevin Gimpel and Karen Livescu},
  booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
  year={2017},
  publisher = {Association for Computational Linguistics}
}


