ml_project
==============================

Production ready project for classification problem. Data is used from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci).

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predictions    <- The final predictions.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Prerequisites

You need to have Python 3.7+ installed in order to run the following project. It's recommended to use a virtual environment to isolate the project's dependencies. One can use conda, virtualenv, pyenv or any other Python environment manager.
I use [virtualenv](https://virtualenv.pypa.io/en/stable/) and [conda](https://conda.io/docs/user-guide/install/index.html).

In order to create an isolated virtual environment and install all necessary packages, you can use the following steps:

1. Create a virtual environment. 
```
python -m venv ./venv
```
2. Activate created virtual environment.

| Platform | Shell           | Command to activate virtual environment |
| -------- | --------------- |-----------------------------------------|
| POSIX    | bash/zsh        | $ source \<venv\>/bin/activate          |
|          | fish            | $ source \<venv\>/bin/activate.fish     |
|          | csh/tcsh        | $ source \<venv\>/bin/activate.csh      |
|          | PowerShell Core | $ source \<venv\>/bin/activate.ps1      |
| Windows  | cmd.exe         | C:\> \<venv\>\Scripts\activate.bat      |
|          | PowerShell      | PS C:\> \<venv\>\Scripts\Activate.ps1   |

3. Install dependencies
```
pip install -r requirements.txt
```

4. Create .env file if it doesn't exist with the following content:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```
Fill the placeholders with your Kaggle credentials.


## How to run pipeline

In order to run the full pipeline execute
```
python src/pipeline/run_pipeline.py
```
This will
- download the data from Kaggle
- transform the data into features
- perform train-test split
- train a model
- evaluate the model on the test set
- save model, metrics and predictions to disk

## How to run webapp
Execute
```
python src/webapp/run_webapp.py
```
This will start the webapp on the localhost:8000. After this one can make a request with
```
python src/webapp/make_request.py
```
to check if the model is working.


## Run webapp tests
```
pytest test/webapp/test_webapp.py
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
