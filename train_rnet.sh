nohup /home/nirvana/miniconda3/bin/python /home/nirvana/projects/python_workspace/cv_projects/mtcnn_in_tf2/train.py \
--stage rnet \
--data_path data/rnet_records/training_data \
--save_path weights \
--learning_rate 1e-4 \
--optimizer adam \
--epochs 60 \
--batch_size 256 \
> rnet.log 2>&1 &
