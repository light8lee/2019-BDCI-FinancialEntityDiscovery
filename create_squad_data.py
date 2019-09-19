# https://gitlab.com/snowhitiger/speakerextraction/blob/master/label_honglou.txt
# coding: utf-8

# In[8]:

import random
import json
import pandas as pd
import os
import numpy as np
import re


# In[9]:
random.seed(2019)
MAX_SEQ_LEN = 128


train_data = pd.read_csv('./data/Train_Data.csv', sep=',', dtype=str, encoding='utf-8')
test_data = pd.read_csv('./data/Test_Data.csv', sep=',', dtype=str, encoding='utf-8')


# In[10]:


train_data.columns


# In[11]:


test_data.columns


# In[12]:


train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)


# In[13]:


img = re.compile(r'\{IMG:\d{1,}\}')
img2 = re.compile(r'<!--(IMG[_\d\s]+)-->')
time = re.compile(r'(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})')
tag = re.compile(r'<(\d|[a-z".A-Z/]|\s)+>')
ques = re.compile(r'[?#/]+')
vx = re.compile(r'(v\d+)|(微信:\d+)')
user = re.compile(r'@.*:')
url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
plain = re.compile(r'\s+')
num = re.compile(r'\d+')
emoji = re.compile(r"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)


def clean(text):
    text = text.replace('&nbsp;', ' ')
    text = url.sub('', text)
    text = emoji.sub('', text)
    text = plain.sub(' ', text)
    text = img.sub('', text)
    text = img2.sub('', text)
    text = time.sub('，', text)
    text = tag.sub('', text)
    text = ques.sub('', text)
    text = vx.sub('', text)
    text = user.sub('', text)
    text = num.sub('0', text)
    return text


# In[14]:


train_data['cleaned_text'] = train_data['text'].apply(clean)
test_data['cleaned_text'] = test_data['text'].apply(clean)
train_data['cleaned_title'] = train_data['title'].apply(clean)
test_data['cleaned_title'] = test_data['title'].apply(clean)
test_data['unknownEntities'] = ''


# In[15]:


train_data = train_data.sample(frac=1, random_state=2019).reset_index(drop=True)
dev_data = train_data.tail(100)
train_data = train_data.head(train_data.shape[0]-100)


# In[16]:


train_data


# In[17]:


# dev_data


# In[18]:


test_data


# In[19]:


def create_tags(text, entities):
    for entity in entities:
        start_pos = text.find(entity)
        if start_pos != -1:
            yield (start_pos, entity)
        


# In[21]:


def merge_sub_texts(sub_texts):
    new_sub_texts = []
    last = []
    curr_len = 0
    for sub_text in sub_texts:
        if curr_len + len(sub_text) < MAX_SEQ_LEN:
            last.append(sub_text)
            curr_len += len(sub_text)
        else:
            if not last:
                new_sub_texts.append(sub_text)
                last = []
                curr_len = 0
            else:
                new_sub_texts.append('。'.join(last))
                last = [sub_text]
                curr_len = len(sub_text)
    if last:
        new_sub_texts.append('。'.join(last))
    return new_sub_texts


comma_stop = re.compile(r'[。]+')
def create_data(data, output_filename, is_test):
    line = 0
    datas = []
    i = 0
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, text, title, entities in zip(data['id'], data['cleaned_text'], data['cleaned_title'], data['unknownEntities']):
            # print('---------------line:', line)
            entities = entities.split(';')
            sub_texts = comma_stop.split(text)
            sub_texts = merge_sub_texts(sub_texts)
            title = title.strip()
            if title:
                sub_texts.append(title)
            print(sub_texts)

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                sub_text = sub_text.replace(' ', '※')
                if not sub_text:
                    continue
                answers = []
                for pos in create_tags(sub_text, entities):
                    answer = {
                        "answer_start": pos[0],
                        "text": pos[1]
                    }
                    answers.append(answer)
                if not answers:
                    continue
                para_entry = dict()
                para_entry["context"] = sub_text
                qas = [{
                    "answers": answers,
                    "question": "有哪些金融实体？",
                    "id": '{}-{}'.format(idx, i)
                }]
                para_entry["qas"] = qas
                data = {
                    "title": "金融实体",
                    "paragraphs": [para_entry]
                }
                datas.append(data)
                i += 1
    outputs = {
        "data": datas,
        "version": "chinese_squad_v1.0"
    }
    with open(output_filename, 'w') as f:
        f.write(json.dumps(outputs, ensure_ascii=False))


# In[22]:


create_data(train_data, 'inputs/train.json', False)


# In[23]:


create_data(dev_data, 'inputs/dev.json', False)


# In[24]:


create_data(test_data, 'inputs/test.json', True)