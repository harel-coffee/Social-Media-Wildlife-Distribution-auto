# NLP-based Classification approach for identifying genuine wildlife observations on Twitter
 Code for the paper "Identifying Wildlife Observations on Twitter", submitted to Environmental Modelling & Software

## Environment

* pip install -r requirements.txt

## Run duplicates removal method
* python findDupTweets.py

## Run feature extraction and feature integration methods
* python wildlifeEmbeddingVec.py


## Run classification methods
* python simpleClassifTwitter.py
* python wildlifeBERT.py
* python wildlifeFastText.py