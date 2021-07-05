from utils import *
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def convert_labels(ner_data):
    new_data = []
    for text, annot in tqdm(ner_data):
        entities = annot['entities']
        new_annot = {'entities': [[s[0], s[1], "DATASET", s[3]] for s in entities]}
        new_data.append((text, new_annot))

    return new_data

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

def convert_ner_data_to_spacy_bin(ner_data, output_path="./train.spacy"):
    for text, annot in tqdm(ner_data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label, _ in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents # label the text with the ents
        db.add(doc)
        
    if output_path:
        db.to_disk(output_path)
        
    return ner_data

selected_pos_ner_data = read_ndjson("selected_pos_ner_data.ndjson")
selected_neg_ner_data = read_ndjson("selected_neg_ner_data.ndjson")

dev_data = selected_neg_ner_data + selected_pos_ner_data
dev_data = convert_labels(dev_data)
convert_ner_data_to_spacy_bin(dev_data, "dev.spacy")

augment = True
train_data = read_ndjson("clean_pos_ner_default_lg_data.ndjson")
outname = "train.spacy"
if augment:
    aug_train_data = read_ndjson("augmented_ner_data.ndjson")
    train_data += aug_train_data
    outname = "train_aug.spacy"
train_data = convert_labels(train_data)

convert_ner_data_to_spacy_bin(train_data, outname)
