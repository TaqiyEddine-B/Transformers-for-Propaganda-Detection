This repository contains the code and resources associated with the paper titled "HTE at ArAIEval Shared Task: Integrating Content Type Information in Binary Persuasive Technique Detection".


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Citation](#citation)

# Introduction
The MARBERT model was fine-tuned for two tasks: identifying persuasive techniques in text (primary binary classification) and classifying text types (tweets or news - auxiliary task). Leveraging the imbalance in the dataset, focal loss was employed for optimization. The system achieved the highest ranking on the leaderboard during testing.

# Installation
The following code blocks outline the steps for environment setup and dependency installation.

```bash
# Clone the repository
git clone https://github.com/TaqiyEddine-B/Transformers-for-Propaganda-Detection.git
cd Transformers-for-Propaganda-Detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#Usage
The input data is expected to be in the `data/` directory. The expected format is a jsonl file with the following structure:

```json
{"id": "0", "text": "This is a sample text.", "label": 1}
{"id": "1", "text": "This is another sample text.", "label": 0}
```

The following code blocks outline the steps for running your code:
```
# Run the main training script
python main.py
```


# Citation
If you use this code in your research, please consider citing the following paper:

```
@article{hte2023,
  title={HTE at ArAIEval Shared Task: Persuasion techniques detection: an interdisciplinary approach to identifying and counteracting manipulative strategies},
  author={Khaldi, Hadjer and Bouklouha, Taqiy Eddine},
  booktitle={Proceedings of the First Arabic Natural Language Processing Conference (ArabicNLP 2023)},
  year={2023},
  address={Singapore},
  organization={Association for Computational Linguistics}
}
```