import os

from torchtext import data

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import senteval
import numpy as np

from torchtext.datasets import SNLI
from torchtext.vocab import GloVe

from model.NLINet import NLINet

import logging

def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.argmax(preds, dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum()
    return acc

def eval_SNLI(NLINet_model,test_iters,args):
    logging.info("**************Start SNLI evaluation*******************")
    eval_acc = 0
    for j, batch in enumerate(test_iters):
        hypothesis = batch.hypothesis
        premise = batch.premise
        labels = batch.label-1

        # forward + backward + optimize
        preds = NLINet_model(hypothesis, premise)
        eval_acc += binary_acc(preds, labels).item()
        print(f"test, step:{j}/{int(len(test_iters.dataset.examples)/args.batch_size_val)}")

    eval_acc = eval_acc/len(test_iters.dataset.examples)
    logging.info(f"SNLI evalution_accuracy: {eval_acc:.5f}")
    print(f"SNLI evalution_accuracy: {eval_acc:.5f}")
    logging.info("***************SNLI evaluation completed********************")
    print('SNLI evaluation completed')



def eval_SentEval(NLINet_model,args,vocabulary,text_field,device):
    logging.info("**************Start SentEval evaluation*******************")

    def batcher(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        torch.cuda.empty_cache()
        with torch.no_grad():
            batch_pad = params.text_field.process(batch,device=params.device)
            batch, batch_len = batch_pad
            embedding_glove = params.NLINet_model.embedding(batch)
            embeddings_sents = params.NLINet_model.encoder_model(embedding_glove,batch_len)
            embeddings_sents = embeddings_sents.cpu().numpy()
        return embeddings_sents

    # parameters
    params = {'task_path': args.PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                            'tenacity': 5, 'epoch_size': 4}

    params["word2id"] = vocabulary.stoi
    params["word_vec"] = vocabulary.vectors
    params["wvec_dim"] = args.embedding_dim
    params["NLINet_model"] = NLINet_model
    params["text_field"] = text_field
    params["device"] = device

    se = senteval.engine.SE(params, batcher)
    transfer_tasks = ['CR','MR', 'SUBJ', 'MPQA', 'SST2', 'TREC',
                      'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
    print("Start SentEval")
    results = se.eval(transfer_tasks)

    print("SentEval evalution_results:", results)
    logging.info(f"SentEval evalution_results: {results}")
    print('SentEval evaluation completed')
    logging.info("***************SentEval evaluation completed********************")





def main(args):
    if args.eval_type == "Both":
        log_file = os.path.join(args.output_dir,"output_eval_SNLI_and_SentEval.log")
    else:
        log_file = os.path.join(args.output_dir,f"output_eval_{args.eval_type}.log")
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s')
    # build dataset and word embedding
    glove = GloVe(name='840B', dim=args.embedding_dim, cache="./dataset/.vector_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #set up fields
    text_field = data.Field(tokenize='spacy',tokenizer_language="en_core_web_sm",
                            lower=True,include_lengths=True,batch_first=True)
    label_field = data.Field(sequential=False)

    train, val, test = SNLI.splits(text_field, label_field, root="./dataset/.data")

    # build vocab
    text_field.build_vocab(train, vectors=glove)
    label_field.build_vocab(train)
    vocabulary = text_field.vocab

    train_iters,val_iters,test_iters = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size_val, device=device)

    # load model
    model_path = os.path.join(args.output_dir, "models", 'best_checkpoint.pkl')
    NLINet_model=torch.load(model_path)
    NLINet_model.eval()

    # evaluation
    if args.eval_type=='SNLI':
        eval_SNLI(NLINet_model,test_iters,args)
    elif args.eval_type=='SentEval':
        eval_SentEval(NLINet_model,args,vocabulary,text_field,device)
    elif args.eval_type=='Both':
        eval_SNLI(NLINet_model,test_iters,args)
        eval_SentEval(NLINet_model,args,vocabulary,text_field,device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluation',
        description='evaluate model')

    parser.add_argument('--embedding_dim', default=300, type=int,
                        help="the dimension of the vocab's embedding")

    parser.add_argument('--batch_size_train', default=64, type=int,
                        help="the batch size of the train phase")

    parser.add_argument('--batch_size_val', default=256, type=int,
                        help="the batch size of val phase")

    parser.add_argument('--lr', default=0.1, type=float,
                        help="learning rate")

    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help="learning rate weight decay")

    parser.add_argument('--hidden_dim_LSTM', default=2048, type=int,
                        help="the dimension of hidden layer in LSTM")

    parser.add_argument('--output_dim', default=3, type=int,
                        help="the dimension of output layer")

    parser.add_argument('--epochs', default=100, type=int,
                        help="the number of epoch")


    parser.add_argument('--encoder_flag', default="BiLSTM", type=str,
                        help="indicate which encoder we choose, it can be one of \
                              ['Baseline', 'UniLSTM', 'SimpleBiLSTM', 'BiLSTM' ]")

    parser.add_argument("--fc_dim", type=int, default=512, help="the dimension of fc layer")

    parser.add_argument('--output_dir', default="./output/BiLSTM", type=str,
                        help="output path")

    parser.add_argument('--eval_type', default="Both", type=str,
                        help="evaluation types, it should be one of ['SNLI','SentEval','Both']")

    parser.add_argument('--PATH_TO_DATA', default="./SentEval/data", type=str,
                        help="the path to the data of SentEval")


    args = parser.parse_args()

    main(args)
