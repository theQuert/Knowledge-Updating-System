# Obtain Vicuna Weights
### Install `FastChat`

```BASH
pip install urllib3==1.26.15
pip install fschat
```

### Clone the `FastChat`, then install it

```BASH
git clone https://github.com/lm-sys/FastChat
cd FastChat/
pip install --upgrade pip
pip install -e .
```

### Download the LLaMA weights (7B / 13B) 
- You need the LLaMA weights in advanced.
- Essentials: `13B.zip`, `tokenizer_check.chk`, `tokenizer.model`
- GDrive id for essentials: `1JZHKkCwPjHeT-G1nC7jWIzTOLDmr5sei`, `1Zvmted1fyjQZexs3CKJ_lglPr0M4jZqS`, `1o3qeWnkMsntE2ZXMp5CHssahUYN2TN9d`

```
mkdir unconverted_weights/
cd unconverted_weights/
gdown id_of_13B.zip
gdown id_of_tokenizer_check.chk
gdown id_of_tokenizer.model
unzip ./13B.zip
```

- Convert the LLaMA weights to HF format

```BASH
mkdir llama_13b_hf
cd FastChat/
pip install protobuf==3.20.0
python ./convert_llama_weights_to_hf.py
--input_dir ../unconverted_weights --model_size 13B --output_dir ../llama_13b_hf
```

- Create the Vicuna Weights (LLaMA + delta = Vicuna)

```BASH
mkdir vicuna_13b
python3 -m fastchat.model.apply_delta \
    --base ./llama_13b_hf \
    --delta lmsys/vicuna-13b-delta-v1.1 \
    --target ./vicuna_13b
```

- Call Vicuna! (with 1 GPU)

```BASH
python3 -m fastchat.serve.cli --model-path ./vicuna_13b
```

- Call Vicuna! (with multiple GPUs)

```BASH
python3 -m fastchat.serve.cli --model-path ./vicuna_13b --num-gpus 2ormore
```

### Requirments (for 13B model)
- GPU: A100 (40GB)
- Memory: 60GB as least
