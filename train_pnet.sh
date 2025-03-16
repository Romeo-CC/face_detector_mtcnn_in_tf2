nohup python /home/nirvana/projects/python_workspace/cv_projects/mtcnn_in_tf2/train.py \
--stage pnet \
--data_path data/pnet_records/training_data \
--save_path weights \
--learning_rate 1e-9 \
--optimizer adam \
--epochs 30 \
--batch_size 512 \
--init_weight_path weights/pnet_r1/30.keras \
> pnet.log 2>&1 &

