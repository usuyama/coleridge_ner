import pandas as pd
import json
import random
import re
from utils import *
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pos_ner_data = read_ndjson("pos_ner_default_lg_data.ndjson")
print(len(pos_ner_data))

new_ng_list = list(set([
    'census bureau',
    'data appendix',
    'data file user',
    'supplementary data',
    'data supplement',
    'major field of study',
    'http',
    'monitor',
    'html',
    'bureau of',
    'data file',
    '†',
    'survey documentation',
    'census of',
    'figure', 'table', 'note', 'tool', 'theory', 'team', 'analyst',
    'data fig', 'fig.', 'tab.', 'journal', 'linear', 'regression',
    'data center', 'office of', 'dataprocess', 'grants program', 'partnership for',
    ' an ', ' a ', 'data export procedure', 'association', '"', "''"
    'study group', 'analyzer', 'integrated system', 'appendix', 'supplementary', 'authority', "inc.", 'ltd.', 'system for', 'system'
    ] + ng_list))

from string import digits

def remove_numbers(text):
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)

def check_ending(text):
    ng_list = ['of', 'the', 'in', 'by', "'s",
     'system', 'systems', 'form', 'committee', 'office', 'analysis', 'equation', 'review panel', 'center', 'program', "'", 
     'training', 'board', 'assistant', 'integration', 'index', 'institute', 'unviersity', 'solutions', 'department', 'project']
    for w in ng_list:
        if text.endswith(w):
            return False

    return True

def check_start(text):
    text = delete_the(text)
    ng_list = ['department', 'university', 'center', 'organization', 'a', 'an', 'division', 'classification']
    for w in ng_list:
        if text.startswith(w):
            return False

    return True

new_data = []
new_data_single = []
new_data_multi = []
for d in pos_ner_data:
    sent = d[0]
    ents = d[1]['entities']
    new_ents = []
    if len(sent) < 50:
        logger.debug("short sent %d, %s", len(sent), sent)
        continue
    if len(sent) > 500:
        logger.debug("long sent %d, %s", len(sent), sent)
        continue

    for e in ents:
        start, end, tag, text = e
        text = text.strip()
        if len(text.split()) == 1:
            continue

        text_wo_number = remove_numbers(text)
        if len(text_wo_number.split()) == 1:
            continue

        preceding_text = sent[start-20:start].lower()
        #preceding_list = ['data']
        following_text = sent[end:end+20].lower()
        #following_list = ['data', 'cohort', 'study']
        surround_ng_list = ['section', 'committee', 'proposal', 'viewed by ', 'conducted by ' 'random', 'consortium', 
        ' an ', ' a ', 'method', 'categor', 'score', 'secretary', 'department', 'rating']
        #good_preceding = detect_keywords(preceding_text, preceding_list)
        #good_following = detect_keywords(following_text, following_list)
        bad_following = detect_keywords(following_text, surround_ng_list)
        bad_preceding = detect_keywords(preceding_text, surround_ng_list)
        # ends with number?
        no_capital = check_no_capital(text)
        detected = re.search(data_like_pattern, text, re.IGNORECASE)
        ng_detected = detect_keywords(text.lower(), new_ng_list)
        good_ending = check_ending(text.lower())
        good_start = check_start(text.lower())
        is_dataset = (tag == 'DATASET')
        if not no_capital and not ng_detected and detected and start != 0 and good_ending and len(text.split()) > 2 and good_start and (not bad_following) and (not bad_preceding):
            new_ents.append((start, end, tag, text))
        elif is_dataset and not no_capital and start != 0 and good_ending and good_start:
            new_ents.append((start, end, tag, text))
        #elif not bad_following and (good_following or good_preceding) and not ng_detected and good_start and len(text.split()) > 2:
        #    print((start, end, tag, text))
        #    new_ents.append((start, end, tag + "_sur", text))

    new_record = (sent, {'entities': new_ents})
    entity_texts = [e[-1] for e in new_ents]
    if len(entity_texts) != len(set(entity_texts)):
        # skip if repeated in same sentence
        # print(new_record)
        continue

    if len(new_ents) > 0:
        new_data.append(new_record)

    if len(new_ents) == 1:
        new_data_single.append(new_record)

    if len(new_ents) > 1:
        new_data_multi.append(new_record)

write_ndjson(new_data, "clean_pos_ner_default_lg_data.ndjson")
write_ndjson(new_data_single, "single_clean_pos_ner_default_lg_data.ndjson")
write_ndjson(new_data_multi, "multi_clean_pos_ner_default_lg_data.ndjson")

print(len(new_data_single), len(new_data_multi))
