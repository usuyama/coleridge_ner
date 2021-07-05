# step1: create weak labels for fine-tuning NER model
# weak labeling sources: (1) acronym detection (2) the government database (3) spacy's pretrained NER model
python create_weak_ner_data.py

# clean up detected NER labels using various filters
python clean_ner_data.py

# convert ndjson files to spacy binary data
python prepare_spacy_binary_data.py

# train spacy NER
# spacy-configs/config_trf.cfg for RoBERTa
python -m spacy train spacy-configs/config.cfg -o trained_model --paths.train ./train.spacy --paths.dev ./dev.spacy --training.patience 1000  --gpu-id 0 --verbose