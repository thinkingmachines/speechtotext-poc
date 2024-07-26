# Ratchada-Whisper
## Overview
Ratchada-Whisper is a project aimed at fine-tuning an open-source Whisper model for Thai speech-to-text conversion. Whisper is a state-of-the-art transformer model known for its high accuracy and low latency in transcribing speech signals into text.

![logo](data/imgs/ratchada_logo.png)

## Key Features

- Fine-tuning of Hugging Face's Whisper implementation
- Custom dataset utilization for Thai audio recordings and transcripts
- GPU infrastructure for efficient training
- Tensorboard integration for monitoring and visualizing the training process
- Comprehensive evaluation of model performance

## Architecture
The project leverages Google Cloud Platform (GCP) components:

![tools](data/imgs/tools.jpg)

- Cloud Storage for raw and processed audio files
- Cloud Functions for preprocessing and post-processing
- Dataflow for feature extraction
- Cloud Run for model inference
- Cloud SQL for storing transcriptions

## Getting Started
### Prerequisites

- Python 3.10
- Poetry for dependency management
- Access to GCP account
- GitHub account with repository access

### Installation

1. Clone the repository:
```zsh
git clone https://github.com/your-org/Ratchada-Whisper.git
cd Ratchada-Whisper
```

2. Set up the development environment:
```zsh
poetry env use python3.10
poetry update
poetry install
poetry run pre-commit install
```

3. Set up environment variables:
```zsh
cp .env.example .env
# Edit .env with your GCP details
```

## Usage

1. Preprocess data: python `preprocess.py`
2. Predict model: python `demo-ratchada.ipynb`
3. Evaluate model: python `evaluation/evaluate.py`
4. Run Jupyter notebooks: `jupyter notebook`

## Deployment
The project uses GitHub Actions for CI/CD. Deployments to dev and production environments are automated. Refer to the Deployment Procedure section in the project documentation for detailed steps.

## Development Workflow
We follow a Trunk-Based Development model. Create feature branches from `master`, make changes, and submit pull requests for review before merging.

## Data Flow

1. Raw audio files are uploaded to a GCS bucket.
2. Preprocessing is triggered by Cloud Functions.
3. Feature extraction is performed using Dataflow.
4. The fine-tuned Whisper model processes the features on Cloud Run.
5. Post-processing is applied to the transcribed text.
6. Final transcriptions are stored in Cloud SQL and GCS.

## Troubleshooting
For common issues and their solutions, refer to the Troubleshooting section in the project documentation.

## Contributing
Please read our contribution guidelines before submitting pull requests.

## License
MIT License
Copyright (c) 2024 Ratchada-Whisper

## Contact
For support or queries, open new issue or contact contributors on the team for the assistance.
