echo "----------------------------------------------------"
cd preprocess
echo "run preprocess..."
echo "get train preprocessed file..."
python -u preprocess.py \
    --input_file ../data/hotpot_data/hotpot_train_v1.1.json \
    --preprocessed_file ../data/hotpot_data/hotpot_train_labeled_data_v3.json
echo "get train processed file done and get dev processed file ..."
python -u preprocess.py \
    --input_file ../data/hotpot_data/hotpot_dev_distractor_v1.json \
    --preprocessed_file ../data/hotpot_data/hotpot_dev_labeled_data_v3.json
echo "get dev processed file done"
echo "run preprocess done!"
echo "----------------------------------------------------"