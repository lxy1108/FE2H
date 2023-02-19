## FE2H-pytorch
A pytorch implementation of our ICASSP2023 paper
>**From Easy to Hard: Two-stage Selector and Reader for Multi-hop Question Answering**  
>Xin-Yi Li, Wei-Jun Lei, Yu-Bin Yang
>Accepted by ICASSP 2023

This repo is still under construction. Please feel free to contact us if you have any questions.

Our result has been published on [HotpotQA Leaderboard](https://hotpotqa.github.io/).

### Data Download and Preprocessing
Run the script to download the data, including HotpotQA data and spacy packages.
```commandline
sh scripts/download.sh
```
Preprocess the training and dev sets in the distractor setting:
```commandline
sh scripts/run_preprocess.sh
```
### Selector
The code for our selector is in directory "**selector_src**".
To train our two-stage selector and infer on the dev set: 
```
sh scripts/run_selector_large.sh
```
### Reader
The code for our reader is in directory "**reader_src**".
To train the reader with our two-stage strategy on ALBERT, and evaluate the model on the dev set: 
```
sh scripts/run_reader_large.sh
```