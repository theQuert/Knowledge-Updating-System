BASE_MODEL="vicuna_13b"
LORA_PATH="lora-Vicuna" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=0 # 1: use local model, 0: use huggingface model
TYPE_WRITER=1 # whether output streamly
CUDA_VISIBLE_DEVICES=0 python vicuna_generate.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL \
    --use_typewriter $TYPE_WRITER
