import argparse
import multiprocessing
import torch
"""
Configuration file

"""
def model_config(parser):
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--covec_on', action='store_false')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--fasttext_on', action='store_true')

    # pos
    parser.add_argument('--no_pos', dest='pos_on', action='store_false')
    parser.add_argument('--pos_vocab_size', type=int, default=54)
    parser.add_argument('--pos_dim', type=int, default=12)
    parser.add_argument('--no_ner', dest='ner_on', action='store_false', help='NER_BIO')
    parser.add_argument('--ner_vocab_size', type=int, default=41)
    parser.add_argument('--ner_dim', type=int, default=8)
    parser.add_argument('--no_feat', dest='feat_on', action='store_false')
    parser.add_argument('--num_features', type=int, default=4)
    # q->p
    parser.add_argument('--prealign_on', action='store_false')
    parser.add_argument('--prealign_head', type=int, default=1)
    parser.add_argument('--prealign_att_dropout', type=float, default=0)
    parser.add_argument('--prealign_norm_on', action='store_true')
    parser.add_argument('--prealign_proj_on', action='store_true')
    parser.add_argument('--prealign_bidi', action='store_true')
    parser.add_argument('--prealign_hidden_size', type=int, default=128)
    parser.add_argument('--prealign_share', action='store_false')
    parser.add_argument('--prealign_residual_on', action='store_true')
    parser.add_argument('--prealign_scale_on', action='store_false')
    parser.add_argument('--prealign_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--prealign_activation', type=str, default='relu')

    parser.add_argument('--pwnn_on', action='store_false')
    parser.add_argument('--pwnn_hidden_size', type=int, default=256, help='support short con')

    ##contextual encoding
    parser.add_argument('--contextual_hidden_size', type=int, default=128)
    parser.add_argument('--contextual_cell_type', type=str, default='lstm')
    parser.add_argument('--contextual_weight_norm_on', action='store_true')
    parser.add_argument('--contextual_maxout_on', action='store_true')
    parser.add_argument('--contextual_residual_on', action='store_true')
    parser.add_argument('--contextual_encoder_share', action='store_true')
    parser.add_argument('--contextual_num_layers', type=int, default=2)

    ## mem setting
    parser.add_argument('--msum_hidden_size', type=int, default=128)
    parser.add_argument('--msum_cell_type', type=str, default='lstm')
    parser.add_argument('--msum_weight_norm_on', action='store_true')
    parser.add_argument('--msum_maxout_on', action='store_true')
    parser.add_argument('--msum_residual_on', action='store_true')
    parser.add_argument('--msum_lexicon_input_on', action='store_true')
    parser.add_argument('--msum_num_layers', type=int, default=1)

    # attention
    parser.add_argument('--deep_att_lexicon_input_on', action='store_false')
    parser.add_argument('--deep_att_hidden_size', type=int, default=128)
    parser.add_argument('--deep_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--deep_att_activation', type=str, default='relu')
    parser.add_argument('--deep_att_norm_on', action='store_false')
    parser.add_argument('--deep_att_proj_on', action='store_true')
    parser.add_argument('--deep_att_residual_on', action='store_true')
    parser.add_argument('--deep_att_share', action='store_false')
    parser.add_argument('--deep_att_opt', type=int, default=0)

    # self attn
    parser.add_argument('--self_attention_on', action='store_false')
    parser.add_argument('--self_att_hidden_size', type=int, default=128)
    parser.add_argument('--self_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--self_att_activation', type=str, default='relu')
    parser.add_argument('--self_att_norm_on', action='store_true')
    parser.add_argument('--self_att_proj_on', action='store_true')
    parser.add_argument('--self_att_residual_on', action='store_true')
    parser.add_argument('--self_att_dropout', type=float, default=0.1)
    parser.add_argument('--self_att_drop_diagonal', action='store_false')
    parser.add_argument('--self_att_share', action='store_false')

    # query summary
    parser.add_argument('--query_sum_att_type', type=str, default='linear',
                        help='linear/mlp')
    parser.add_argument('--query_sum_norm_on', action='store_true')

    parser.add_argument('--max_len', type=int, default=5)
    parser.add_argument('--decoder_num_turn', type=int, default=5)
    parser.add_argument('--decoder_mem_type', type=int, default=1)
    parser.add_argument('--decoder_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--decoder_opt', type=int, default=0)
    parser.add_argument('--decoder_att_hidden_size', type=int, default=128)
    parser.add_argument('--decoder_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--decoder_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--decoder_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--decoder_weight_norm_on', action='store_true')
    parser.add_argument('--classifier_merge_opt', type=int, default=0)
    parser.add_argument('--classifier_dropout_p', type=float, default=0.4)
    parser.add_argument('--classifier_weight_norm_on', action='store_false')
    parser.add_argument('--classifier_gamma', type=float, default=1)
    parser.add_argument('--classifier_threshold', type=float, default=0.5)
    parser.add_argument('--label_size', type=int, default=1)

    ### ELMo setting
    parser.add_argument('--elmo_on', action='store_true')
    parser.add_argument('--elmo_config_path', default='data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
    parser.add_argument('--elmo_weight_path', default='data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')
    parser.add_argument('--elmo_size', type=int, default=1024)
    parser.add_argument('--elmo_lexicon_on', action='store_true')
    parser.add_argument('--elmo_att_on', action='store_false')
    parser.add_argument('--elmo_self_att_on', action='store_true')
    parser.add_argument('--elmo_l2', type=float, default=0.001)
    parser.add_argument('--elmo_dropout', type=float, default=0.5)

    return parser

def data_config(parser):
    parser.add_argument('--v2_on', action='store_true')
    parser.add_argument('--log_file', default='san.log', help='path for log file.')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--meta', default='meta')
    parser.add_argument('--train_data', default='train_data',
                        help='path to preprocessed training data file.')
    parser.add_argument('--dev_data', default='dev_data',
                        help='path to preprocessed validation data file.')
    parser.add_argument('--dev_gold', default='dev',
                        help='path to preprocessed validation data file.')
    parser.add_argument('--test_data', default='test_data',
                        help='path to preprocessed test data file.')
    parser.add_argument('--test_gold', default='test',
                        help='path to preprocessed validation data file.')
    parser.add_argument('--covec_path', default='data/MT-LSTM.pt')
    parser.add_argument('--glove', default='data/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words.'
                             'Otherwise consider question words first.')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                        help='number of threads for preprocessing.')
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='Use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=100)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', default='adamax')
    parser.add_argument('--grad_clipping', type=float, default=5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_cov', type=float, default=0.4)
    parser.add_argument('--dropout_w', type=float, default=0.05)
    # scheduler
    parser.add_argument('--no_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--fix_embeddings', action='store_true', help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--tune_partial', type=int, default=1000, help='finetune top-x embeddings (including <PAD>, <UNK>).')
    parser.add_argument('--model_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018)

    return parser

def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args
