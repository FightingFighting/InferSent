# 导入torchtext相关包
import os

import numpy as np
from torchtext import data

import argparse
import torch

from torchtext.datasets import SNLI
from torchtext.vocab import GloVe

def main(args):

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

    # load model
    model_path = os.path.join(args.model_dir, "models", 'best_checkpoint.pkl')
    NLINet_model = torch.load(model_path)
    NLINet_model.eval()

    sentence = args.sentence.split(" ")
    if sentence[-1][-1:] == ".":
        sentence[-1] = sentence[-1][0:-1]
        sentence.append(".")
    sentence = [sentence]

    torch.cuda.empty_cache()
    with torch.no_grad():
        sentence = text_field.process(sentence,device=device)
        sentence, sentence_len = sentence
        embedding_glove = NLINet_model.embedding(sentence)
        embeddings_sents = NLINet_model.encoder_model(embedding_glove,sentence_len)
        embeddings_sents = embeddings_sents.cpu().numpy()

    os.makedirs(args.feature_dir)
    save_path = os.path.join(args.feature_dir,"extracted_feature")
    np.save(save_path,embeddings_sents)



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

    parser.add_argument("--fc_dim", type=int, default=512, help="the dimension of fc layer")

    parser.add_argument('--model_dir', default="./output/Baseline", type=str,
                        help="output path")

    parser.add_argument('--sentence', default="a woman is making music.", type=str,
                        help="hypothesis sentence")

    parser.add_argument('--feature_dir', default="./feature_dir", type=str,
                        help="hypothesis sentence")

    args = parser.parse_args()

    main(args)
