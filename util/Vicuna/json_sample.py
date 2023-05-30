dataset_data = [
    {
        "instruction": "",
        "input": ,
        "output": 
    }
    for idx in range(len(ids))
]


# Dump our data to json format
with open("./ptfile/vicuna_lora/merge.json", "w") as f:
    json.dump(dataset_data, f)
