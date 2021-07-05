import pandas as pd
import json
import random
import re
from utils import *
import spacy
from spacy import displacy

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

blank_nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

def convert_doc(text, annot):
    doc = blank_nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label, _ in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents

    return doc

def visualize_ner(ner_data, output_path="vis.html", limit=100, shuffle=False):
    htmls = []
    rrange = list(range(len(ner_data)))
    if shuffle:
        random.shuffle(rrange)
    for i in rrange:
        d = ner_data[i]
        doc = convert_doc(d[0], d[1])
        print(doc)

        html = displacy.render(doc, style="ent")
        htmls.append(html + "-----<br>")    

        if len(htmls) > 100:
            break

    with open(output_path, 'w') as f:
        f.write("\n".join(htmls))

if __name__ == '__main__':
    pos_ner_data = read_ndjson("multi_clean_pos_ner_default_lg_data.ndjson")
    print(len(pos_ner_data))

    visualize_ner(pos_ner_data)
