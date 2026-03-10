# CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-12.4/bin/nsys profile -t cuda,osrt,nvtx --force-overwrite=true -o ./out/ectract_128 /mnt/data0/home/syt/FlyHE/build/bin/conver/example/extract

CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-12.4/bin/nsys \
 profile -t cuda,osrt,nvtx --force-overwrite=true \
 -o ./out/repack_64 \
 /mnt/data0/home/syt/FlyHE/build/bin/conver/example/repack
