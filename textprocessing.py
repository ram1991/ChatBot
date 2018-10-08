import json
import os
import numpy as np
import pandas as pd

json_files = []

for each_json_file in os.listdir(files_path):
    if each_json_file.endswith('.json'):
        json_files.append(each_json_file)

input_texts = []
target_texts = []

for each_file in json_files:
    file = json.load(open(each_file, 'r'))
    text_data = []
    for dictionary in file:
        temp_list = []
        for dialog in dictionary['dialog']:
            temp_list.append((dialog['sender'], dialog['text']))
        text_data.append(temp_list)
    for each_sent in text_data:
        for each_message in each_sent:
            #print(each_message)
            if each_message[0] == 'participant1':
                #print(each_message)
                input_texts.append(each_message[1])
            elif each_message[0] == 'participant2':
                target_texts.append(each_message[1])
 

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize 
import string
from string import punctuation
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
english_words = set(nltk.corpus.words.words())
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 


def tokenize(text):
    
    tokenized_list = []

    for each_sent in text: 
       # print(each_sent)
       temp_sent_list= []
       sentences = sent_tokenize(each_sent)
       temp_sent_list.append(sentences)
       #print(sentences)
       for each_word in sentences:
           temp_word_list = []
           if each_word not in stop_words and each_word not in string.punctuation:
                if each_word in english_words or not each_word.isalpha():
                    #print(each_word)
                    words = word_tokenize(each_word.lower())
                    temp_word_list.append(words)
                    #words = re.sub('\d+','',str(words))
                    #words = re.sub('[.,;?]','',str(words))
                    tokenized_list.append(words)
    return tokenized_list

