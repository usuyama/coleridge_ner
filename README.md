# Weakly-supervised NER fine-tuning for Coleridge dataset detection

This is a part of our [4th place winning solution](https://www.kaggle.com/osciiart/210622-det1-neru-train-govt) for the [Coleridge Initiative Kaggle competition](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/).

A typical approach for this competition is to train a named entity recognition (NER) model to detect dataset mentions. However, one of the biggest challenges in this competition is the scarcity of training labels. Instead of adding more annotations manually, weakly supervised learning generates training labels programmatically by combining heuristics, ontology-based string matching, pretrained-models, &c. 

In this competition, we combined the following labeling functions to generate training labeles:
1. acronym detection
2. the government database
3. spacy's pretrained NER model

We simply combine the generated labels and use the union as training labels. (Other ways of combining labeling sources might be worth exploring e.g. soft labels, majority vote, &c.)
In addition, we apply keyword/rule-based filters to clean up generated labels and reduce false-positive labels.

We take the spacy's pretrained NER model `en_core_web_lg-3.0.0` and fine-tune it using our generated labels.

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

The trained model will saved to `trained_model/model-best`

# Inference

Load the trained model as spacy pipeline:
```
nlp = spacy.load("trained_model/model-best")
```

* Example notebook: https://www.kaggle.com/naotous/run-finetuned-ner-offline
* Winning solution: https://www.kaggle.com/osciiart/210622-det1-neru-train-govt



