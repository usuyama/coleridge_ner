# %%

# Asthetics
import logging
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Basic
import pandas as pd
#pd.set_option('display.max_columns', 10000)
import numpy as np
import json
import os
import random
from tqdm.autonotebook import tqdm
import string
import re
from functools import partial
from pprint import pprint
import ast
tqdm.pandas()
import spacy
print(spacy.__version__)
import logging
logger = logging.getLogger(__name__)
from utils import *

def write_ndjson(data, output_path):
    with open(output_path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

def read_ndjson(input_path):
    data = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))
            
    return data   

# %% 

df_labels = pd.read_csv("data/df_labels_210619_01.csv")
print(df_labels.shape)
print(df_labels)

patterns = []
for row in df_labels.iterrows():
    label = row[1].label
    pattern = {"label": "DATASET", "pattern": label}
    patterns.append(pattern)
    # too noisy...
    # pattern = {"label": "DATASET", "pattern": row[1].acronym_clean}
    # patterns.append(pattern)

print(len(patterns))
write_ndjson(patterns, "patterns.ndjson")

from spacy.lang.en import English

ruler_nlp = English()
config = {
   "phrase_matcher_attr": 'LOWER',
}
ruler = ruler_nlp.add_pipe("entity_ruler", config=config)
ruler.add_patterns(patterns)

sent = "This study used data from the National Education Longitudinal Study (NELS:88) to examine the effects of dual enrollment programs for high school students on college degree attainment."
doc = ruler_nlp(sent)
entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents]
print(entities)

# %%


df_papers = pd.read_csv('data/train_papers.csv')
df_papers['text_list'] = df_papers['text_list'].progress_apply(lambda x: ast.literal_eval(x))
print(df_papers.shape)
print(df_papers)


filter_keywords = ['study', 'report', 'program', 'analy', 'based', 'according', 'census', 'initiative', 'data', 'benchmark', 'survey', 'sample', 'procedure', 'result', 'studies', 'test', 'obtain']
filter_pattern = '|'.join(filter_keywords)

#%%

import spacy
print("gpu", spacy.require_gpu())
nlp_sent = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
nlp_sent.add_pipe('sentencizer')


# %%

pos_ner_data = {}
neg_ner_data = {}
for i in tqdm(range(len(df_papers))):    
    text_list = df_papers.iloc[i]['text_list']
    for text in text_list:
        if is_long_text(text):
            continue
        for sent_obj in nlp_sent(text).sents:
            sent = sent_obj.text.strip()
            if re.search(filter_pattern, sent, re.IGNORECASE):
                sent_doc = ruler_nlp(sent)
                entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in sent_doc.ents]
                d = (sent, {'entities': entities})
                if len(entities) > 0:
                    pos_ner_data[sent] = d
                else:
                    neg_ner_data[sent] = d

    if i % 100 == 0:
        write_ndjson(list(pos_ner_data.values()), "pos_ner_data.ndjson")
        write_ndjson(list(neg_ner_data.values()), "neg_ner_data.ndjson")

write_ndjson(list(pos_ner_data.values()), "pos_ner_data.ndjson")
write_ndjson(list(neg_ner_data.values()), "neg_ner_data.ndjson")

print(len(pos_ner_data))
print(len(neg_ner_data))

# %%
text_list
# %%
