# Get Vicuna and Finetune with Our Data!
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

- Call Vicuna in interactive mode! (with 1 GPU)

```BASH
python3 -m fastchat.serve.cli --model-path ./vicuna_13b
```

- Call Vicuna in interactive mode! (with multiple GPUs)

```BASH
python3 -m fastchat.serve.cli --model-path ./vicuna_13b --num-gpus 2ormore
```

### Finetune and Inference (LLaMA, Vicuna)
- Essentials: **Data in json format**, *scripts from [here](https://github.com/theQuert/NetKUp/tree/main/util/Vicuna)*
- The weights are fed in HF format.
- `finetune_with_lora.py`: modify the path in **L27-L29**, then run it.
- `vicuna_generate.py`: modify the path as well in **L16-L17**.
- `util/generate.py`: run it.
- Prepare data in json format: [Reference](https://github.com/theQuert/NetKUp/blob/main/util/Vicuna/json_sample.py)
- Call Vicuna in api mode (vanilla and fine-tuned Vicuna)

```BASH
# you have to modify the path in L17, L18, L210, L214 first
python3 ./util/Vicuna/vicuna_generate.py
```

### Requirments (for 13B model)
- GPU: A100 (40GB)
- Memory: 60GB as least
