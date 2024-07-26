# Overview
This project aims to fine-tune an open-source Whisper model for (Thai!) speech to text task on open source Whisper model.

Whisper is a state-of-the-art transformer model that can transcribe speech signals into text with high accuracy and low latency. We will use the huggingface's whisper implementation to fine-tune the model on our own GPU infrastructure, using a various custom dataset of audio recordings and transcripts.

We will also monitor the training process and evaluate the model performance with tensorboard, a visualization tool for machine learning experiments.

The tools used in this repository for finetuning can be described below:

![tools](data/imgs/tools.jpg)


## Setup dev environment
poetry env use python3.10
poetry update
poetry install
poetry run pre-commit install
