import argparse


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_config():
    """ 模型配置 """
    parser = argparse.ArgumentParser()
    # 必须参数
    # 模型参数
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese."
                        )
    parser.add_argument("--over_write_result", default=True, type=bool,
                        help="over write the result")
    parser.add_argument("--no_network", type=str2bool, default=False)
    parser.add_argument("--checkpoint_path", default='../../data/pretrained_models/albert-xxlarge-v2', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--output_dir", default='../checkpoints/selector/second_hop_selector', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--feature_cache_path", default="../data/cache/selector/second_hop_selector")
    parser.add_argument("--model_name", type=str, default='BertForRelated',
                        help="The output directory where the model checkpoints and predictions will be written.")
    # 数据输入
    parser.add_argument("--log_prefix", default="20211209_bert_selector")
    parser.add_argument("--train_file", default='../data/hotpot_data/hotpot_train_labeled_data.json', type=str,
                        help="train_file")
    parser.add_argument("--first_predict_result_path", default="../data/selector/first_hop_result/", type=str,
                        help="The output directory of all result")
    parser.add_argument("--best_paragraph_file", default='train_best_paragraph.json', type=str,
                        help="best_paragraph_file")
    parser.add_argument("--related_paragraph_file", default='train_related_paragraph.json', type=str,
                        help="related_paragraph_file")
    parser.add_argument("--new_context_file", default='train_new_context.json', type=str,
                        help="new_context_file")
    parser.add_argument("--dev_best_paragraph_file", default='dev_best_paragraph.json', type=str,
                        help="best_paragraph_file")
    parser.add_argument("--dev_related_paragraph_file", default='dev_related_paragraph.json', type=str,
                        help="related_paragraph_file")
    parser.add_argument("--dev_new_context_file", default='dev_new_context.json', type=str,
                        help="new_context_file")
    parser.add_argument("--dev_file", default='../data/hotpot_data/hotpot_dev_labeled_data.json', type=str,
                        help="SQuAD json for evaluation. ")
    # 其他参数
    parser.add_argument("--log_path", default="../../log", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=256, type=int)
    parser.add_argument("--use_file_cache", default=True, type=bool,
                        help="use the feature cache or not")
    parser.add_argument("--sent_overlap", default=2, type=int,
                        help="When splitting up a long document into chunks, "
                             "how much sentences is overlapped between chunks.")
    parser.add_argument("--train_batch_size", default=24, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=128, type=int, help="Total batch size for validation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--save_model_step',
                        type=int, default=5000,
                        help="The proportion of the validation set")
    return parser