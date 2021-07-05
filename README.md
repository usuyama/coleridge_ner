# Weakly-supervised NER fine-tuning for Coleridge dataset detection

Weak supervision sources
(1) acronym detection (2) the government database (3) spacy's pretrained NER model

spacy NER pretrained model: en_core_web_lg-3.0.0

# Requirements

```
spacy==3.0.6
```

# Training

```sh
# step1: create weak labels for fine-tuning NER model
# weak labeling sources: (1) acronym detection (2) the government database (3) spacy's pretrained NER model
python create_weak_ner_data.py

# step2: clean up detected NER labels using various filters
python clean_ner_data.py

# step3: convert ndjson files to spacy binary data
python prepare_spacy_binary_data.py

# step4: train spacy NER
# spacy-configs/config_trf.cfg for RoBERTa
python -m spacy train spacy-configs/config.cfg -o trained_model --paths.train ./train.spacy --paths.dev ./dev.spacy --training.patience 1000  --gpu-id 0 --verbose

# step5: quick inspection/visualization of the trained model
python try_trained_model.py
```

# Inference

TODO: link notebook

