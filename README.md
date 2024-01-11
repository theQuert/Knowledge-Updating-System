# Full-Text Level Knowledge Updating System

<p align="center">
✍️ <a href="http://140.119.164.212:7840" target="_blank"><b>Online Demo</b></a> 
•
🤗 <a href="https://huggingface.co/theQuert" target="_blank"><b>HF Repo</b></a>
•
📃 <a href="https://dl.acm.org/doi/10.1145/3511808.3557537" target="_blank"><b>Paper</b></a>  
•
📎 <a href="https://github.com/theQuert/Event-Triggered-Article-Updating-System/tree/main/docs/YTLee_s_Oral_Presentation.pdf" target="_blank"><b>Presentation</b></a>
•
🗒️ <a href="https://drive.google.com/file/d/1RzfgyGdtvYvein9Z34xLVSlGLViz_UH9/view?usp=sharing" target="_blank"><b>Master's Thesis</b></a>

## Overview
**Event Triggered Article Updating System** is a long article updating application for knowledge update.

Spearheaded the development of an Event Triggered Article Updating System, a cutting-edge application designed for updating long articles with new knowledge. This project showcases a significant advancement in handling full-text knowledge updating triggered by news events, leveraging the capabilities of large language models (LLMs).

The whole system is trained on [**NetKu dataset**](https://github.com/hhhuang/NetKu) for *knowledge updating in full-text triggered by a News Event*.

## Demo
A live demonstation of the model can be accessed at [Live Demo](http://140.119.164.212:7840) with GPU support, and [HF Space](https://huggingface.co/spaces/theQuert/Event-Triggered-Article-Updating-System) with CPU support.

## Key Features
1. **Long texts input support**: Overcame the limitations of existing LLMs by enabling the system to understand and process long context inputs. Developed a unique approach allowing for unlimited full-article input lengths, with each paragraph handling up to 4,096 tokens.

2. **Instruction-Tuned Models**: Implemented multiple baseline models, including those fine-tuned on LLaMA, Alpaca, Vicuna, and GPT-based models, demonstrating versatility and adaptability in model training.

3. **Innovative Model Architecture**: Proposed and developed a new Encoder-Decoder based model architecture. Conducted comprehensive evaluations to prove the effectiveness of this novel approach in the context of knowledge updating.


## Citations
If you use our code, data, or models in your research, please cite this repository. You can use the following BibTeX entry:

```bibtex
@inproceedings{lee2022multi,
  title={A Multi-grained Dataset for News Event Triggered Knowledge Update},
  author={Lee, Yu-Ting and Tang, Ying-Jhe and Cheng, Yu-Chung and Chen, Pai-Lin and Li, Tsai-Yen and Huang, Hen-Hsen},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4158--4162},
  year={2022}
}
```

## License
The code in this project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## OpenAI Data Acknowledgment
The text generation included in this project were generated using OpenAI's models and are subject to OpenAI's Terms of Use. Please review [OpenAI's Terms of Use](https://openai.com/policies/terms-of-use) for details on usage and limitations.

## Acknowledgements
This work is supported by 
1. [**National Science and Technology Council, Taiwan**](https://www.nstc.gov.tw/?l=en), under grants `109-2222-E-001-004-MY3` and `109-2628-H-004-001-MY4`.
2. [**Institute of Information Science, Academia Sinica, Taiwan**](https://www.iis.sinica.edu.tw/en/index.html/).
3. [**National Chengchi University, Taiwan**](https://www.nccu.edu.tw/).
4. We thank [**Meta LLaMA team**](https://github.com/facebookresearch/llama), [**Vicuna team**](https://github.com/lm-sys/FastChat), [**Lightning AI**](https://lightning.ai/) and [**ISI-NLP**](https://github.com/isi-nlp/NewsEdits) for their contributions.

