import os

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
    vocabulary_label = label_field.vocab.itos


    # load model
    model_path = os.path.join(args.model_dir, "models", 'best_checkpoint.pkl')
    NLINet_model = torch.load(model_path)
    NLINet_model.eval()

    hypothesis = args.hypothesis.split(" ")
    premise = args.premise.split(" ")
    if hypothesis[-1][-1:] == ".":
        hypothesis[-1] = hypothesis[-1][0:-1]
        hypothesis.append(".")
    if premise[-1][-1:] == ".":
        premise[-1] = premise[-1][0:-1]
        premise.append(".")

    hypothesis = [hypothesis]
    premise = [premise]
    hypothesis = text_field.process(hypothesis,device=device)
    premise = text_field.process(premise,device=device)
    preds = NLINet_model(hypothesis, premise)
    preds_argmax = torch.argmax(preds,dim=1)
    preds_label = vocabulary_label[preds_argmax+1]
    print(preds_label)

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


    parser.add_argument('--hypothesis', default="a woman is making music.", type=str,
                        help="hypothesis sentence")

    parser.add_argument('--premise', default="a pregnant lady singing on stage while holding a flag behind her.", type=str,
                        help="premise sentence")

    args = parser.parse_args()

    main(args)
