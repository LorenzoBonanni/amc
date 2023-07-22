python amc_search.py 
    --job=train 
    --model=efficentnet_b4 
    --dataset=imagenet 
    --preserve_ratio=0.5 
    --lbound=0.2 
    --rbound=1 
    --reward=acc_reward 
    --data_root=./imagenet 
    --ckpt_path=./checkpoints/efficentnet_b4_imagenet.pth.tar 
    --seed=2018
