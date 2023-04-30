echo 'start pretraining'
work_dir='./output/snclr/exp_xxx'

python3 main_log.py \
  --dist-url tcp://$main_ip:13848 \
  --multiprocessing-distributed --world-size 4 --rank $local_rank \
  -a vit_small -b 4096 \
  --optimizer=adamw --lr=1.5e-5 --weight-decay=0.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 \
  --snclr-m-cos \
  --snclr-t=.2 \
  --snclr-m 0.99 \
  --script start.sh \
  --dist-backend gloo \
  --workdir $work_dir \
  --ssl-type 'snclr_factor' \
  --bank-size 128000 --topk 30 --bank-epoch 0 \
  ./data/ILSVRC/data/