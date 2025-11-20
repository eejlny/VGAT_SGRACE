# VGAT\_SGRACE



Model that combines a CNN backbone and GNN layers to perform image classification.



GATConv 



currently 0.91 accuracy on CIFAR10 with run:



python3 desktop\_cifar\_graph\_train.py --device cuda --amp auto --tf32 --epochs 200 --batch-size 128 --backbone cnn --embed-dim 128 --hidden 256 --heads 8 --num-blocks 3 --lr 0.001 --weight-decay 0.001 --warmup-epochs 5 --label-smoothing 0.1 --knn 0 --edge-drop 0 --mixup 0 --cutmix 0 --randaug --ra-n 2 --ra-m 9 --random-erasing 0.1 --ema 0 --tta --early-stop 0



0.8986 accuracy with 1 head and 1 block :



python3 desktop\_cifar\_graph\_train.py --device cuda --amp auto --tf32 --epochs 200 --batch-size 128 --backbone cnn --embed-dim 128 --hidden 128 --heads 1 --num-blocks 1 --lr 0.001 --weight-decay 0.001 --warmup-epochs 5 --label-smoothing 0.1 --knn 0 --edge-drop 0 --mixup 0 --cutmix 0 --randaug --ra-n 2 --ra-m 9 --random-erasing 0.1 --ema 0 --tta --early-stop 0



GCNConv



0.8879



python3 desktop\_cifar\_graph\_train.py --device cuda --amp auto --tf32 --epochs 200 --batch-size 128 --backbone cnn --embed-dim 128 --hidden 128 --heads 1 --num-blocks 1 --lr 0.001 --weight-decay 0.001 --warmup-epochs 5 --label-smoothing 0.1 --knn 0 --edge-drop 0 --mixup 0 --cutmix 0 --randaug --ra-n 2 --ra-m 9 --random-erasing 0.1 --ema 0 --tta --early-stop 0



SAGEConv



0.8980



python3 desktop\_cifar\_graph\_train.py --device cuda --amp auto --tf32 --epochs 200 --batch-size 128 --backbone cnn --embed-dim 128 --hidden 128 --heads 1 --num-blocks 1 --lr 0.001 --weight-decay 0.001 --warmup-epochs 5 --label-smoothing 0.1 --knn 0 --edge-drop 0 --mixup 0 --cutmix 0 --randaug --ra-n 2 --ra-m 9 --random-erasing 0.1 --ema 0 --tta --early-stop 0





0.845



python3 desktop\_cifar\_graph\_train.py --device cuda --amp auto --tf32 --epochs 200 --batch-size 128 --backbone cnn --embed-dim 128 --hidden 16 --heads 1 --num-blocks 1 --lr 0.001 --weight-decay 0.001 --warmup-epochs 5 --label-smoothing 0.1 --knn 0 --edge-drop 0 --mixup 0 --cutmix 0 --randaug --ra-n 2 --ra-m 9 --random-erasing 0.1 --ema 0 --tta --early-stop 0





GATv2Conv







