# Development Lifecycle
## Trunk Based Development

![tools](../data/imgs/tools.jpg)

The set-speechtotext-poc project follows the concept of Trunk-based Development, wherein User Stories are worked on PRs. PRs then get merged to master once approved by the team.
The master branch serves as the most up-to-date version of the code base. For releases, whenever we deploy, we create a `release/<version>` branch, which is based off of `master` at a given point in time, and serves as the branch to be deployed.

## Naming Format

### Branch Names:

#### To `dev` Branch:
- `feature/<issue-id>-<header>`
- `fix/<issue-id>-<header>`


#### To master Branch:

- `release/<version>`
- `release/<version>-<hotfix>`



### PR Title:
`[<Feature/Fix/Release/Hotfix>](<issue-id>) <Short desc>`

### PR Template:
`pull_request_template.md`

## Development Workflow

1. Create a new branch from `master` for your feature or fix
2. Make changes locally and push to the remote branch
3. Create a Pull Request to merge your branch into `master`
4. After code review and approval, merge the PR
5. For releases, create a `release/<version>` branch from `master`

## Local Development
### File Structure

- docs/ - Contains all Markdown files for project documentation
- notebooks/ - Jupyter notebooks for data exploration, model training, and evaluation
  - preprocessing/ - Scripts for audio data preprocessing
  - models/ - Implementation of speech recognition models
  - evaluation/ - Scripts for model evaluation
- requirements.txt - Python dependencies

## Pre-requisites

- Access to the project's GCP account
- Access to the Bitwarden collection for secrets
- GitHub account with access to the project repository
- Python 3.8 or higher
- Jupyter Notebook

### Cloning and Installation

1. Clone the repository:
```zsh
git clone https://github.com/your-org/set-speechtotext-poc.git
cd set-speechtotext-poc
```

2. Create a virtual environment:
```zsh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```zsh
pip install -r requirements.txt
```

### Environment Setup

1. Copy the `.env` if applicable:
```zsh
cp .env.example .env
```
2. Fill in the required environment variables in the `.env` file:
```zsh
GCP_PROJECT_ID=your-project-id
GCP_BUCKET_NAME=your-bucket-name
MODEL_VERSION=1.x.x
```

## Running the Application

1. ctivate the virtual environment if not already active:
```zsh
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. To preprocess data:
```zsh
python src/preprocessing/preprocess.py
```

3. To train the model:
```zsh
python src/models/train.py
```

4. To evaluate the model:
```zsh
python src/evaluation/evaluate.py
```

5. To run Jupyter notebooks:
```zsh
jupyter notebook
```
Navigate to the `notebooks/` directory and open the desired notebook.

Remember to update the `requirements.txt` file whenever you add new dependencies to the project.
