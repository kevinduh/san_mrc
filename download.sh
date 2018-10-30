#!/usr/bin/env bash
############################### 
# Download data resources
# including SQuAD v2.0 
# by xiaodl@microsoft.com
############################### 

DATA_DIR=$(pwd)/data
echo $DATA_DIR
mkdir $DATA_DIR

## Download SQuAD data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $DATA_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $DATA_DIR/dev-v1.1.json

## download SQuAD v2.0 data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O $DATA_DIR/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $DATA_DIR/dev-v2.0.json

# Download GloVe
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $DATA_DIR/glove.840B.300d.zip
unzip $DATA_DIR/glove.840B.300d.zip -d $DATA_DIR
rm $DATA_DIR/glove.840B.300d.zip

# Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O $DATA_DIR/MT-LSTM.pt

# Download ELMo
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 -O $DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json -O $DATA_DIR/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
