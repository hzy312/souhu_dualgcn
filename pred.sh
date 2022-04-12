python predict.py \
--batch_size 64 \
--gpus -1 \
--predict_data_path ./data_preprocess/test_processed.txt \
--state pred \
--predict_ckpt_path ./save/last.ckpt