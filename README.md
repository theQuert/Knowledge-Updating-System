# News Event Triggered Knowledge Update Generation

## Processing 
- Run sentence-level difference encoding with `netku_edit.py`
- Conbine the labeled non-updated information with the updated contents: `netku_edit_post.py` to re-construct the labeled pt-format file
- Create special-token-added tokenizer with the script: `primer/script/primer_toks.py`
- Using our custom tokenizer to truncate the max sequence length to meet the **PRIMERA** requirements: *input<=4096* with *output<=1024*: `netku_edit_beftrunc.py`
- Run script to re-construcure the input data to maximum 10 paragraphs in each list from each cluster (WCEP10-like): `netku_edit_afttrunc.py`

## Source of our dataset
#### [Data](https://drive.google.com/drive/folders/1xF05auIp1EMd_KMHEaEu48J9meOsZSrU?usp=sharing)

## Bi-directional Sentence Labeling
![bidirectional_labeling](https://raw.githubusercontent.com/theQuert/NetKu_Processing/main/bi_labeling.png)

## References
#### [A Multi-grained Dataset for News Event Triggered Knowledge Update](https://github.com/hhhuang/NetKu)
#### [PRIMERA](https://github.com/allenai/PRIMER)
#### [NewsEdits](https://github.com/isi-nlp/NewsEdits)
