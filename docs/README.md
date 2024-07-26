# Ratchada-Whisper
## Project Overview
Ratchada-Whisper is a proof-of-concept project for speech-to-text conversion using machine learning models. It processes audio files and converts them into text using a data pipeline built on Google Cloud Platform (GCP).

![logo](../data/imgs/ratchada_logo.png)

## Key Features

- Audio preprocessing and normalization
- Feature extraction using MFCC
- Speech-to-text model inference
- Post-processing of transcribed text
- Scalable architecture using GCP services

## Architecture
The project uses the following GCP components:

- Cloud Storage for raw and processed audio files
- Cloud Functions for preprocessing and post-processing
- Dataflow for feature extraction
- Cloud Run for model inference
- Cloud SQL for storing transcriptions

## Getting Started
### Prerequisites

- Python 3.10 or higher
- Access to GCP account
- GitHub account with repository access

### Installation

1. Clone the repository:
```zsh
git clone https://github.com/your-org/Ratchada-Whisper.git
cd Ratchada-Whisper
```

2. Set up a virtual environment:
```zsh
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```zsh
pip install -r requirements.txt
```

4. Set up environment variables:
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
We follow a Trunk-Based Development model. Create feature branches from master, make changes, and submit pull requests for review before merging.

## Troubleshooting
For common issues and their solutions, refer to the Troubleshooting section in the project documentation.

## Contributing
Please read our contribution guidelines before submitting pull requests.

## License
MIT License

## Contact
For support or queries open new issue or contact the maintainers.
