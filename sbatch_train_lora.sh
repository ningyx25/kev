#!/bin/bash
#SBATCH --job-name=qwen3.5-9B-lora-jev
#SBATCH --output=/home/ningyongxin/workplace/proj/kev/log.txt
#SBATCH --error=/home/ningyongxin/workplace/proj/kev/err.txt
#SBATCH --time=144:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4

PROJECT=/home/ningyongxin/workplace/proj/kev
cd "$PROJECT"
source .venv/bin/activate

VISION_DATA=data/processed/ac-jev-v2-subset100/train.jsonl
VISION_BASE=./Qwen/Qwen3.5-9B
OUT_DIR=runs/clef-v2-qwen35-9B-lora
SWANLAB=kev-vision

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node 4 -m kev.train \
  --vision_data "$VISION_DATA" \
  --vision_base "$VISION_BASE" \
  --clef 1 \
  --lora 16 --weights_dtype bf16 --dtype bf16 \
  --device cuda \
  --checkpointing 1 \
  --max_state 8192 \
  --max_image_pixels $((1024*1024)) \
  --max_history_pixels $((512*512)) \
  --lr 2e-5 --epochs 3 \
  --save_every_steps 500 \
  --log_every 1 \
  --swanlab "$SWANLAB" \
  --out "$OUT_DIR"
#   --resume 1
