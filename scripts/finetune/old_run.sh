#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
# export TRITON_CACHE_PATH=/home/nilanb/nilanb_ada/users/nilanb/yer/.triton
# export CUDA_HOME=/cm/shared/apps/cuda10.2
# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# export LD_LIBRARY_PATH=$CUDA_HOME/toolkit/10.2.89/lib64:$CUDA_HOME/toolkit/10.2.89/extras/CUPTI/lib64:/usr/ebuild/software/GCCcore/11.2.0/lib64:/cm/local/apps/gcc/10.2.0/lib:/cm/local/apps/gcc/10.2.0/lib64:/cm/local/apps/gcc/10.2.0/lib32:/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64:$CUDA_HOME/toolkit/10.2.89/include
# export LIBRARY_PATH=$CUDA_HOME/toolkit/10.2.89/lib64:/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64
MODEL_PATH='/home/pratiky1/nilanb_ada/users/pratiky1/fact/VideoMAEv2/models/vit_g_hybrid_pt_1200e_k710_ft.pth'
OUTPUT_DIR='/home/pratiky1/nilanb_ada/users/pratiky1/VideoMAEv2/output/pratik'
DATA_PATH='/home/pratiky1/nilanb_ada/users/pratiky1/VideoMAEv2/dataset/UCF101_subset_basketball/'

# adding variable DATA_ROOT for custom dataset
DATA_ROOT='/home/pratiky1/nilanb_ada/users/pratiky1/VideoMAEv2/dataset/UCF101_subset_basketball'
# 
# --data_root ${DATA_ROOT} \

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
#         --job-name=${JOB_NAME} \
#         --gres=gpu:${GPUS_PER_NODE} \
#         --ntasks=${GPUS} \
#         --ntasks-per-node=${GPUS_PER_NODE} \
#         --cpus-per-task=${CPUS_PER_TASK} \
#         --kill-on-bad-exit=1 \
        # --async \
        # ${SRUN_ARGS} \
 #        --data_root ${DATA_ROOT} \
# adding data_root after --model
        python run_class_finetuning_copy01.py \
        --model vit_base_patch16_224 \
        --data_set UCF101Subset \
        --nb_classes 2 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 2 \
        --opt adamw \
        --lr 7e-4 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 50 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
