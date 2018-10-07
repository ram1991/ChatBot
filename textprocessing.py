import json
import os
import numpy as np
import pandas as pd
json_files = []

input_texts = []
target_texts = []

for each_json_file in os.listdir(files_path):
    if each_json_file.endswith('.json'):
        json_files.append(each_json_file)


for each_file in json_files:
    file = json.load(open(each_file, 'r'))
    text_data = []
    for dictionary in file:
        temp_list = []
        for dialog in dictionary['dialog']:
            temp_list.append((dialog['sender'], dialog['text']))
        text_data.append(temp_list)
    for each_line in text_data:
        for each_message in each_line:
            #print(each_message)
            if each_message[0] == 'participant1':
                #print(each_message)
                input_texts.append(each_message[1])
            elif each_message[0] == 'participant2':
                target_texts.append(each_message[1])
                
