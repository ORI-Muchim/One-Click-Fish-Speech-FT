# One-Click-Fish-Speech-FT

### [Original Repo](https://github.com/fishaudio/fish-speech)

Multilingual Speech Synthesis based on Fish-Speech

- Supported Language: English, Japanese, Korean, Chinese, French, German, Arabic, and Spanish

## Table of Contents 
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Preprocess](#preprocess)
- [Fine-Tuning](#fine-tuning)
- [Inference](#inference)
- [References](#references)

## Prerequisites
- A Windows/Linux system with a minimum of `16GB` RAM.
- A GPU with at least `12GB` of VRAM.
- Python == 3.10
- Anaconda installed.
- PyTorch installed.
- CUDA installed.

Pytorch install command:
```sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

CUDA 11.7 install:
`https://developer.nvidia.com/cuda-11-7-0-download-archive`

---

## Installation 
1. **Create an Anaconda environment:**

```sh
conda create -n oneclickfish python=3.10
```

2. **Activate the environment:**

```sh
conda activate oneclickfish
```

3. **Clone this repository to your local machine:**

```sh
git clone https://github.com/ORI-Muchim/One-Click-Fish-Speech-FT.git
```

4. **Navigate to the cloned directory:**

```sh
cd One-Click-Fish-Speech-FT
```

5. **Install the necessary dependencies:**

```sh
pip install -r requirements.txt
```

---

## Preprocess

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```
You need to convert your dataset into the above format and place it under data. The audio file can have the extensions .mp3, .wav, or .flac, and the annotation file should have the extensions .lab.

---

## Fine-Tuning

To start this tool, use the following command, replacing {data-dir} and {project-name} with your respective values:

```sh
python finetune.py --data-dir {data-dir} --project-name {project-name}
```

---

## Inference

After the model has been trained, you can inferred by using the following command:

```python
python inference.py --text "The text you want to convert" --checkpoint-dir "checkpoints/fish-speech-1.5" --reference-audio "{your_audio}.wav" --prompt-text "{Transcript of reference-audio}" --project-name {your-project-name}
```

---

## References

For more information, please refer to the following: 
- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}
```
