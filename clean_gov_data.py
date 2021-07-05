import pandas as pd
import os
import sys
import re
from utils import *
from tqdm.autonotebook import tqdm
tqdm.pandas()

from titlecase import titlecase

df = pd.read_csv('data/govt_much_bigger_datasets.csv')

print(df)

prefix_pattern = r"([A-Z]+[0-9]+[A-Z]*) (.*)"

def clean_text(text):
    text = text.replace('\n', '').replace("_", " ").encode('ascii', 'ignore').decode()
    text = re.sub('\s+',' ', text).strip()
    text = text.rstrip(string.digits)
    m = re.match(prefix_pattern, text)
    if m:
        text = m.group(1)

    return text

new_ng_list = list(set([
    'census bureau',
    'data appendix',
    'data file user',
    'supplementary data',
    'data supplement',
    'major field of study',
    'http',
    'html',
    'data file',
    '†',
    'figure', 'table', 'note', 'tool', 'theory', 'team', 'analyst',
    'data fig', 'fig.', 'tab.',
    '"', "''"
    'study group', 'analyzer', 'integrated system', 'appendix', 'supplement', 'authority', "inc.", 'ltd.', 'system for', 'system'
    ] + ng_list))

df['title'] = df['title'].progress_apply(lambda x: clean_text(x))
df['len'] = df['title'].progress_apply(lambda x: len(x.split()))
df['datalike'] = df['title'].progress_apply(lambda x: bool(re.search(data_like_pattern, x, re.IGNORECASE)))
df['ng'] = df['title'].progress_apply(lambda x: detect_keywords(x.lower(), new_ng_list))
df['title_titlecase'] = df['title'].progress_apply(lambda x: titlecase(x))
print(df)
print(df.shape)

df2 = df[(df['title_titlecase'] == df['title']) & (df['len'] > 2) & (df['len'] < 8) & df.datalike & (~df.ng)].drop_duplicates()

print(df2.shape)

df2.to_csv('data/govt_much_bigger_datasets_titlecase.csv')
