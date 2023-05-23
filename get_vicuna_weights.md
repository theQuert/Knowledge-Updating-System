# Obtain Vicuna Weights
### Install `FastChat`

```BASH
pip install urllib3==1.26.15'
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
- Convert the LLaMA weights to HF format

```BASH
mkdir ./llama_13b_hf
cd FastChat/
pip install protobuf==3.20.0
python ./convert_llama_weights_to_hf.py
--input_dir ../unconverted_weights --model_size 13B --output_dir ../llama_13b_hf
```

- Create the Vicuna Weights (LLaMA + delta = Vicuna)

```
cd ../
mkdir vicuna_13b
python3 -m fastchat.model.apply_delta \
    --base ./llama_13b_hf \
    --delta lmsys/vicuna-13b-delta-v0 \
    --target ./vicuna_13b
```
