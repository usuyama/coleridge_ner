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


data_like_keywords = ['study', 'report', 'program', 'analy', 'census', 'initiative', 'data', 'benchmark', 'survey', 'sample', 'procedure', 'result', 'studies', 'test', 'system', 'evaluation', 'assessment', 'trend', 'monitor', 'index']
data_like_pattern = '|'.join(data_like_keywords)

filter_keywords = data_like_keywords + ['based', 'according', 'obtain']
filter_pattern = '|'.join(filter_keywords)

import spacy
print("gpu", spacy.require_gpu())
nlp_sent = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
nlp_sent.add_pipe('sentencizer')

def truncate_text(text, limit=1000000):
    logger.warning("long text %d, %s ...", len(text), text[:100])
    return text[:limit]

##
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

nlp_ner = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
#nlp = spacy.load('en_core_web_trf') # RoBERTa
nlp_ner.add_pipe('merge_entities')
nlp_ner.add_pipe('merge_noun_chunks')

config = {
   "phrase_matcher_attr": 'LOWER',
}
ruler = nlp_ner.add_pipe("entity_ruler", config=config)
ruler.add_patterns(patterns)

def detect_entities(text):
    doc = nlp_ner(text)

    return (text, [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents])

# def find_conjunct_noun_chunks(text):
#     doc = nlp_ner(text)
#     chunks = list(doc.noun_chunks)
    
#     conjunct_groups = set()
    
#     entities = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
#     print("named_entities", entities)
    
#     for chunk in chunks:
#         #print(type(chunk.root), chunk.root.i)
#         #print(type(chunk.root.head), chunk.root.head.i)
#         #print(chunk.text, list(chunk.noun_chunks), chunk.start, chunk.end, chunk.root.text, chunk.root.dep_, chunk.root.head.text, chunk.conjuncts)
#         if len(chunk.conjuncts) > 0:
#             group = tuple(sorted([chunk.text] + [s.text for s in chunk.conjuncts]))
#             conjunct_groups |= {group}
            
#     return conjunct_groups

sample_text = "A number of longitudinal epidemiologic studies, including the Baltimore Longitudinal Study of Aging, the New Mexico Aging Process Study, and the Massachusetts Male Aging Study, have demonstrated age-related increases in the likelihood of developing hypogonadism."
print(detect_entities(sample_text))


df_papers = pd.read_csv('data/train_papers.csv')
df_papers['text_list'] = df_papers['text_list'].progress_apply(lambda x: ast.literal_eval(x))
print(df_papers.shape)
print(df_papers)

target_ner_labels = ['ORG', 'WORK OF ART', 'EVENT', "DATASET"]

pos_ner_data = {}
neg_ner_data = {}
import random
randomrange = list(range(len(df_papers)))
random.shuffle(randomrange)
for i in tqdm(randomrange):    
    text_list = df_papers.iloc[i]['text_list']
    for text in text_list:
        if is_long_text(text):
            continue

        for sent_obj in nlp_sent(text).sents:
            sent = sent_obj.text.strip()
            if re.search(filter_pattern, sent, re.IGNORECASE):
                sent_doc = nlp_ner(sent)
                entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in sent_doc.ents if ent.label_ in target_ner_labels]
                d = (sent, {'entities': entities})
                if len(entities) > 0:
                    pos_ner_data[sent] = d
                else:
                    neg_ner_data[sent] = d

    if i % 100 == 0:
        write_ndjson(list(pos_ner_data.values()), "pos_ner_default_lg_data.ndjson")
        #write_ndjson(list(neg_ner_data.values()), "neg_ner_default_lg_data.ndjson")

write_ndjson(list(pos_ner_data.values()), "pos_ner_default_lg_data.ndjson")
#write_ndjson(list(neg_ner_data.values()), "neg_ner_default_lg_data.ndjson")
