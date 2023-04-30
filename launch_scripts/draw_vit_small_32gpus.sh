echo 'start pretraining'
workdir='./train_output/snclr/vit/debug_neighbor05'
echo tcp://$2:12347
echo node:$2 --- rank:$1

python3 main_draw_neighbors.py \
  -a vit_small -b 4096 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=0.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 \
  --moco-m-cos \
  --moco-t=.2 \
  --dist-url tcp://$2:15345 \
  --multiprocessing-distributed --world-size 4 --rank $1 \
  --script vit_small_32gpus.sh \
  --dist-backend gloo \
  --workdir $workdir \
  --script vit_small_32gpus.sh \
  --ssl-type 'snclr_draw_neighbors' \
  --moco-m 0.99 \
  --bank-size 128000 --topk 30 --bank-epoch 0 \
  ./0_public_datasets/imageNet_2012/

