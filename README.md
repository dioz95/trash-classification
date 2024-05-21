# Automated Trash Type Classification
This repository contains experiment to build Convolutional Neural Network (CNN) based model to classify the waste material based on the images data available on this [Hugging Face Repository](https://huggingface.co/datasets/garythung/trashnet). The model is built using the `dataset-rezized` dataset for the sake of training efficiency.

## Project Structures
The project is published in 3 main files:
1. `Exploratory Image Analysis and Model Experimentation.ipynb` : This notebook contains initial exploration of the data and several experiments to develop the model. Two models are generated -- 1 model trained using augmented training dataset and 1 model training without augmented training data.
2. `Result Analysis`: This notebook compares the performance of the two models generated in the first notebook file using the validation data.
3. `train_model.py` : This python script automates the model development. Simple hyperparameters are used to simplify the model training. Metrics tracking and model versioning are performed using [Weight & Biases (Wandb)](https://wandb.ai/adamata-selection/wandb-trash-classification?nw=nwuseradvendiodesandros1) in the `wandb-trash-classification` project.

## Automated Training
The training script `train_model.py` will run automatically at every 12 AM and when a commit is pushed to the main branch. The detailed automation script is written in `.github/workflows/automate_train.yml` by using `WANDB_API_KEY` as a secret variable.

## How to Reproduce
To replicate the experimentation process you will need to:
1. Clone the repository,
```bash
git clone <github-repository-url>
```
2. Create environment using `python3.11.x`. In this case I use conda,
```bash
conda create -n "<env-name>" python=3.11
```
3. Install required dependencies,
```bash
pip install -r requirements.txt
```
4. Create `models` directory in the project root directory,
```bash
mkdir -p `./models/`
```
5. Run `train_mode.py`,
```bash
python train_model.py
```
