
# coding: utf-8

# In[8]:


import pandas as pd
import os
import numpy as np
import re


# In[9]:


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
img2 = re.compile(r'<!--IMG_\d+-->')
time = re.compile(r'(\d{4}-\d{2}-\d{2})|(\d{2}:\d{2}:\d{2})')
tag = re.compile(r'<(\d|[a-z".A-Z/]|\s)+>')
ques = re.compile(r'[?#/]+')
vx = re.compile(r'(v\d+)|(微信:\d+)')
user = re.compile(r'@.*:')
def clean(text):
    text = text.replace('&nbsp;', ' ')
    text = img.sub('', text)
    text = img2.sub('', text)
    text = time.sub('，', text)
    text = tag.sub('', text)
    text = ques.sub('', text)
    text = vx.sub('', text)
    text = user.sub('', text)
    return text


# In[14]:


train_data['cleaned'] = train_data['text'].apply(clean)
test_data['cleaned'] = test_data['text'].apply(clean)
test_data['unknownEntities'] = ''


# In[15]:


train_data = train_data.sample(frac=1, random_state=2019).reset_index(drop=True)
dev_data = train_data.tail(500)
train_data = train_data.head(train_data.shape[0]-500)


# In[16]:


train_data


# In[17]:


dev_data


# In[18]:


test_data


# In[19]:


def findall(text, entity):
    text_length = len(text)
    entity_length = len(entity)
    result = []
    begin = 0
    if not entity:
        return result
    while True:
        pos = text.find(entity, begin)
        if pos != -1:
            result.append((pos, pos+entity_length))
            begin += pos + entity_length
        else:
            return result


# In[20]:


def create_tags(text, entities):
    tags = ['O'] * len(text)
    for entity in entities:
        print('entity:', entity)
        for begin, end in findall(text, entity):
            tags[begin] = 'B'
            for i in range(begin+1, end):
                tags[i] = 'I'
    print(tags)
    return tags
        


# In[21]:


comma_stop = re.compile(r'[。，,]+')
def create_data(data, output_filename):
    line = 0
    with open(output_filename, 'w', encoding='utf-8') as f:
        for idx, text, entities in zip(data['id'], data['cleaned'], data['unknownEntities']):
            print('---------------line:', line)
            entities = entities.split(';')
            
            for sub_text in comma_stop.split(text):
                f.write('^'*10)
                f.write(idx)
                f.write('\n')
                print(sub_text)
                for char, tag in zip(sub_text, create_tags(sub_text, entities)):
                    f.write('{} {}\n'.format(char, tag))
                f.write('$'*10)
                f.write('\n')
            line += 1


# In[22]:


create_data(train_data, 'inputs/train.txt')


# In[23]:


create_data(dev_data, 'inputs/dev.txt')


# In[24]:


create_data(test_data, 'inputs/test.txt')
