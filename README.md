# Event Triggered Article Updating System

<p align="center">
✍️ <a href="http://140.119.164.212:7840" target="_blank">Online Demo</a> 
•
🤗 <a href="https://huggingface.co/theQuert" target="_blank">HF Repo</a>
•
📃 <a href="https://dl.acm.org/doi/10.1145/3511808.3557537" target="_blank">[Paper]</a>  

## Overview
Event Triggered Article Updating System (NetKUp) is a long article updating application for knowledge update.

**NetKUp** trained on [NetKu dataset](https://github.com/hhhuang/NetKu) for knowledge updating task.

## Demo
A live demonstation of the model can be accessed at [Live Demo](http://140.119.164.212:7840) with GPU support, and [HF Space](https://huggingface.co/spaces/theQuert/Event-Triggered-Article-Updating-System) with CPU support.

## Key Features
1. **Long texts input support**: Curently due to the limitation of LLMs, long context input is not clearly understood, and hard to capture relationshtip between pargraphs. We make it suitable for longer inputs.

2. **Instruction-Tuned**: We provide multiple baselines, including fine-tuned our data on **Alpaca**, **Vicuna**,  and GPT based models.

3. **Model Architecture**: We proposed the new construction of Encoder-Decoder based model. And further prove the effectiveness of our model.

## TBD
- [ ]**Paper**

## Citations
If you use our code, data, or models in your research, please cite this repository. You can use the following BibTeX entry:

```bibtext
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
4. We thank [**Meta LLaMA team**](https://github.com/facebookresearch/llama), [**Vicuna team**](https://github.com/lm-sys/FastChat), [**Lightning AI**](https://lightning.ai/)and [**ISI-NLP**](https://github.com/isi-nlp/NewsEdits) for their contributions.

