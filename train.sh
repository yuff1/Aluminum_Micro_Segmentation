CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/unetplusplus.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/deeplabv3plus.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config/knet.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/mobilenet_unet.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/segnet.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/resunet.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config/resunet_ca.py 1