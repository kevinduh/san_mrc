This is a packge of SAN (Stochastic Answer Networks for Machine Reading Comprehension).

# Please cite the following papers if you use this package:
https://arxiv.org/abs/1804.07888
https://arxiv.org/abs/1712.03556


######### SETUP ENV ###########
1. python3.6
2. install requirements:
   >pip install -r requirements.txt
3. download data/word2vec 
   >sh download.sh


###############################

######### Train a SAN Model #####
1. preproces data
   >python prepro.py
2. train a model
   >python train.py
################################

Some of codes are borrowed from: https://github.com/hitvoice/DrQA


by
xiaodl@microsoft.com
