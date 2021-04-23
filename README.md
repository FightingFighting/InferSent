# InferSent
This is a simple implementation of the [paper](https://arxiv.org/abs/1705.02364) (Supervised Learning of Universal Sentence Representations from Natural Language Inference Data), including training the Natural Language Inference and build sentence representation encoder.

Four Natural Language Inference model was implemented:
  * Baseline: Averaging word embeddings to obtain sentence representations.
  * UniLSTM: A LSTM was applied on the word embeddings, where the last hidden state is considered as sentence representation.
  * SimpleBiLSTM: The last hidden state of forward and backward layers of BiLSTM are concatenated as the sentence representations.
  * BiLSTM: The max pooling was applied to the concatenation of word-level hidden states from both directions of BiLSTM to retrieve sentence representations.

# Installation
1. Create your own conda environment(python==3.6.13)
2. Install pytorch and relavant dependencies
  * [pytorch](https://pytorch.org/get-started/previous-versions/) 
    ```
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
    ```
  * tensorboard == 2.5.0
  * torchtext == 0.8.1
  * [spacy](https://v2.spacy.io/usage) == 3.0.5 
    ```
    conda install -c conda-forge spacy
    python -m spacy download en_core_web_sm
    ```
3. Download dataset 
In this code, the [SNLI](https://nlp.stanford.edu/projects/snli/) corpus was used to train Natural Language Inference model. The `torchtext` was used in the code, so when run the `train.py` at first time, the SMLI dataset will be downloaded automatically into folder `./dataset/.data`
4. Download GloVe word vector
when run the `train.py` at first time, the GloVe will be downloaded automatically into folder `./dataset/.vector_cache`
5. In order to evaluate the sentence representation, the [SentEval](https://github.com/facebookresearch/SentEval) was used. please follow the instruction to install and clone it into the folder `SentEval`.

# Usage
## Pretrained model
The pretrained models can be download in [here](https://drive.google.com/drive/folders/1AcdqUqgMFbFGoJYDMmMpyihlngUtuV1b?usp=sharing). Save them into the folder `dataset`.

## Train
For example, training __Baseline__ model. 
```
python train.py --encoder_flag Baseline --output_dir ./output/Baseline
```
The _--encoder_flag_ can be __Baseline__ / __UniLSTM__ / __SimpleBiLSTM__ / __BiLSTM__ .

## Evaluation (SentEval)
For example, evaluating __Baseline__ sentence encoder:
```
python eval.py --encoder_flag Baseline --output_dir ./output/Baseline
```
The _--eval_type_ can be __SNLI__ / __SentEval__ / __Both__ :
  * __SNLI__ : evaluating the Natural Language Inference.
  * __SentEval__ : evaluating sentence encoder using __SentEval__.
  * __Both__ : evaluating above ( __SNLI__ and __SentEval__ ).


## Inferentce (Natural Language Inference -- sentence entailment)
For example, a hypothesis: _a woman is making music._ and a premise _a pregnant lady singing on stage while holding a flag behind her._:
```
python nli.py --model_dir "./output/Baseline" --hypothesis "a woman is making music." --premise "a pregnant lady singing on stage while holding a flag behind her."
```
It will generate the relation class ( _Nature_, _contradiction_ and _entailment_) between hypothesis and premise.

## Extract Sentence Feature
For example, extracting the representation of _a woman is making music._
```
python nli.py --model_dir "./output/Baseline" --sentence "a woman is making music. --feature_dir XXX"
```
It will save the sentence feature into _XXX_.


