#!/bin/bash

download_tfhub_model () {
    curl -L --output $1.tar.gz https://tfhub.dev/tensorflow/$1/$2?tf-hub-format=compressed
    mkdir -p $1
    tar -zxvf $1.tar.gz -C $1
    rm $1.tar.gz
}


download_tfhub_model bert_en_uncased_preprocess 3

download_tfhub_model bert_en_uncased_L-12_H-768_A-12 4


