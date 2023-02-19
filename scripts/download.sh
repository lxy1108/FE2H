# 下载Hotpot数据
# e是判断文件是否存在 -d filename 如果 filename为目录，则为真
if [ ! -e "./data/hotpot_data/hotpot_train_v1.1.json" ]; then
    mkdir -p data/hotpot_data
    cd data/hotpot_data
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
  else
    echo "文件已存在"
fi
# 下载Spacy数据，对名词类词汇进行标准化
pip install spacy
echo "download spacy models"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz
