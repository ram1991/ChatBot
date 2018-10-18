

import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize 
import string
from string import punctuation
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 


tokenized_xtext = []

for each_sent in incoming_message:
#    print(each_sent)
    sentences = sent_tokenize(each_sent)
    print(sentences)
    for each_word in sentences:
        print(each_word)
        if each_word not in stop_words and each_word not in string.punctuation:
            words = word_tokenize(each_word.lower())
            tokenized_xtext.append(words)
        
tokenized_ytext = []

for each_sent in target_message:
#    print(each_sent)
    sentences = sent_tokenize(each_sent)
    print(sentences)
    for each_word in sentences:
        print(each_word)
        if each_word not in stop_words and each_word not in string.punctuation:
            words = word_tokenize(each_word.lower())
            tokenized_ytext.append(words)


counter = 0

for each_sent in tokenized_ytext:
    each_sent.insert(0,'START')
    each_sent.append('END')
    counter += 1
    
complete_text = tokenized_xtext + tokenized_ytext 
   
vocab = {}

for each_sent in complete_text:
    for each_word in each_sent:
        if each_word not in vocab:
            vocab[each_word] = 1
        else:
            vocab[each_word] += 1

token = ['_PAD', '_GO', '_EOS', '_UNK']
            
vocab_list = token + sorted(vocab, key = vocab.get ,reverse = True)


input_vocab = {}

for each_sent in tokenized_xtext:
    for each_word in each_sent:
        if each_word not in input_vocab:
            input_vocab[each_word] = 1
        else:
            input_vocab[each_word] += 1
            
input_vocabulary = sorted(input_vocab, key = input_vocab.get,reverse = True)

input_word2index = {}

for index, word in enumerate(input_vocabulary):
    input_word2index[word] = index

input_index2word = {}

for index,word in input_word2index.items():
    input_index2word[index] = word

target_vocab = {}

for each_sent in tokenized_ytext:
    for each_word in each_sent:
        if each_word not in target_vocab:
            target_vocab[each_word] = 1
        else:
            target_vocab[each_word] += 1
    
target_vocabulary = sorted(target_vocab, key = target_vocab.get, reverse = True)

target_word2index = {}

for index,word in enumerate(target_vocabulary):
    target_word2index[word] = index

target_index2word = {}

for index,word in target_word2index.items():
    target_index2word[index] = word
    
