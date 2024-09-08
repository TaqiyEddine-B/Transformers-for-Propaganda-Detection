This repository contains the code and resources associated with the paper titled "HTE at ArAIEval Shared Task: Integrating Content Type Information in Binary Persuasive Technique Detection".


## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Tracking Experiments](#tracking-experiments)
- [Usage](#usage)
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
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

# Project Structure
The project structure is as follows:
```bash
.
├── data
│   ├── tasklA_dev.jsonl
│   ├── tasklA_test_no_label.jsonl
│   ├── tasklA_test.jsonl
│   ├── tasklA_train_dev.jsonl
│   ├── tasklA_train.jsonl
│   ├── tasklB_dev.jsonl
│   ├── task1B_test_no_label.jsonl
│   └── task1B_train.jsonl
├── src
│   ├── models
│   ├── eval_task1a.py
│   ├── loss.py
│   ├── main_task1A.py
│   ├── main_task1B.py
│   └── utils.py
├── requirements.txt
└── README.md

- `data/`: Directory containing the input data in jsonl format.
- `src/`: Directory containing the source code.
- `src/main_task1A.py`: Main script for training and evaluation of the task1A.
- `src/main_task1B.py`: Main script for training and evaluation of the task1B.
- `requirements.txt`: File containing the dependencies.
- `README.md`: This file.

```

## Data
The input data is expected to be in the `data/` directory. The expected format is a jsonl file with the following structure:

```json
{"id": "0", "text": "This is a sample text.", "label": 1}
{"id": "1", "text": "This is another sample text.", "label": 0}
```

# Tracking Experiments
The `wandb` library is used for tracking experiments.

# Usage
The following code blocks outline the steps for running your code:
```bash
# Run the main training script
python src/main_task1B.py
# or
python src/main_task1A.py
```


# Citation
If you use this code in your research, please consider citing the following paper:

```
@inproceedings{hadjer-bouklouha-2023-hte,
    title = "{HTE} at {A}r{AIE}val Shared Task: Integrating Content Type Information in Binary Persuasive Technique Detection",
    author = "Hadjer, Khaldi  and
      Bouklouha, Taqiy",
    editor = "Sawaf, Hassan  and
      El-Beltagy, Samhaa  and
      Zaghouani, Wajdi  and
      Magdy, Walid  and
      Abdelali, Ahmed  and
      Tomeh, Nadi  and
      Abu Farha, Ibrahim  and
      Habash, Nizar  and
      Khalifa, Salam  and
      Keleg, Amr  and
      Haddad, Hatem  and
      Zitouni, Imed  and
      Mrini, Khalil  and
      Almatham, Rawan",
    booktitle = "Proceedings of ArabicNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.arabicnlp-1.46",
    pages = "502--507",
    abstract = "Propaganda frequently employs sophisticated persuasive strategies in order to influence public opinion and manipulate perceptions. As a result, automating the detection of persuasive techniques is critical in identifying and mitigating propaganda on social media and in mainstream media. This paper proposes a set of transformer-based models for detecting persuasive techniques in tweets and news that incorporate content type information as extra features or as an extra learning objective in a multitask learning setting. In addition to learning to detect the presence of persuasive techniques in text, our best model learns specific syntactic and lexical cues used to express them based on text genre (type) as an auxiliary task. To optimize the model and deal with data imbalance, a focal loss is used. As part of ArabicNLP2023-ArAIEval shared task, this model achieves the highest score in the shared task 1A out of 13 participants, according to the official results, with a micro-F1 of 76.34{\%} and a macro-F1 of 73.21{\%} on the test dataset.",
}
```