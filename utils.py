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

    logger.info("%d rows from %s", len(data), input_path)
            
    return data   

def check_no_capital(x):
    try:
        if x[1:]==x[1:].lower():
            return True
        return False
    except:
        return False

def delete_the(x):
    try:
        x_split = x.split()
        if x_split[0]=='the':
            return ' '.join(x_split[1:])
        return x
    except:
        return x

def detect_acronym(x, ref_label, ref_acronym, ref_target, th=[1,2]):
    try:
        ans = []
        for i in range(len(ref_label)):
            if x.count(ref_label[i])>=th[0] and x.count(ref_acronym[i])>=th[1]:
                ans.append(ref_target[i])
    #             print(i)
        return ans
    except:
        return []
    
def detect_keywords(x, ref):
    try:
        for keyword in ref:
            if keyword in x:
                return True
        return False
    except:
        return False

keywords = [
    'study',
    'studies',
    'data',
    'survey',
    'panel',
    'census',
    'cohort',
    'longitudinal',
    'registry',   
]

keywords2 = [
    'study',
    'studies',
    'data',
    'survey',
    'panel',
    'census',
    'cohort',
    'longitudinal',
    'registry',
    'the',
]
keywords3 = [
    'study',
    'studies',
    'dataset',
    'database',
    'survey',
    'panel',
    'census',
    'cohort',
    'longitudinal',
    'registry',
]
keywords4 = [
    'system',
    'center',
    'centre',
    'committee',
    'documentation',
    'entry',
    'assimilation',
    'explorer',
    'regulation',
    'portal',
    'format',
    'data science',
    'analysis',
    'management',
    'agreement',
    'branch',
    'acquisition',
    'request',
    'task force',
    'program',
    'operator',
    'office',
    'data view',
    'data language',
    'mission',
    'alliance',
    'data model',
    'data structure',
    'corporation',
]

white_list = [
    'ipeds',
]

keywords5 = keywords + ['of', 'the', 'national', 'education']

ng_list = [
    'national longitudinal survey',
    'education longitudinal survey',
    'census bureau',
    'data appendix',
    'data file user',
    'supplementary data',
    'data supplement',
    'major field of study',
    'http',
    'html',
]
black_list = [
    'USGS',
    'GWAS',
    'ECLS',
    'DAS',
    'NCDC',
    'NDBC',
    'UDS',
    'GTD',
    'ISC',
    'DGP',
    'EDC',
    'FDA',
    'TSE',
    'DEA',
    'CDA',
    'IDB',
    'NGDC',
    'JODC',
    'EDM',
    'FADN',
    'LRD',
    'DBDM',
    'DMC',
    'WSC',
    ###count4##
]

data_like_keywords = [
    'study', 'report', 'program', 'analy', 'census', 'initiative', 'data', 'benchmark',
    'survey', 'sample', 'procedure', 'result', 'studies', 'evaluation',
    'assessment', 'trend', 'monitor', 'index',
    'panel',
    'cohort',
    'longitudinal',
    'registry',
    'system',
    'interpolation'
    ]
data_like_pattern = '|'.join(data_like_keywords)

filter_keywords = data_like_keywords + ['based', 'according', 'obtain', 'test']
filter_pattern = '|'.join(filter_keywords)

def truncate_text(text, limit=20000):
    is_long_text(text, limit)
    text[:limit]

def is_long_text(text, limit=20000):
    if len(text) > limit:
        logger.warning("long text %d, %s ... %s", len(text), text[:50], text[-50:])
        return True
    else:
        False