Задание к 9-му модулю. 
Датасет https://www.kaggle.com/competitions/forest-cover-type-prediction

## Выполнение заданий:
1. Use the Forest train dataset. You will solve the task of forest cover type prediction and compete with other participants. (necessary condition, 0 points for the whole homework if not done) - **DONE**
2. Format your homework as a Python package. Use an src layout or choose some other layout that seems reasonable to you, explain your choice in the README file. Don't use Jupyter Notebooks for this homework. Instead, write your code in .py files. (necessary condition, 0 points for the whole homework if not done) - **DONE**
3. Publish your code to Github. (necessary condition, 0 points for the whole homework if not done) - **DONE**. 
3.1. Commits should be small and pushed while you're working on a project (not at the last moment, since storing unpublished code locally for a long time is not reliable: imagine something bad happens to your PC and you lost all your code). Your repository should have at least 30 commits if you do all non-optional parts of this homework. (12 points) - **DONE**
4. Use Poetry to manage your package and dependencies. (6 points) - **DONE**  
4.1. When installing dependencies, think if these dependencies will be used to run scripts from your package, or you'll need them only for development purposes (such as testing, formatting code, etc.). For development dependencies, use the dev option of add command. If you decided not to use Poetry, list your dependencies in requirements.txt and requirements-dev.txt files. (4 points) - **DONE**
5. Create a data folder and place the dataset there. (necessary condition, 0 points for the whole homework if not done. Note for reviewers: data folder won't be seen on GitHub if added to gitignore, it's OK, check gitignore) - **DONE**   
5.1. Don't forget to add your data to gitignore. (5 points) - **DONE**  
5.2. (optional) Write a script that will generate you an EDA report, e.g. with pandas profiling - **DONE**  
6. Write a script that trains a model and saves it to a file. Your script should be runnable from the terminal, receive some arguments such as the path to data, model configurations, etc. To create CLI, you can use argparse, click (as in the demo), hydra, or some other alternatives. (10 points) - **DONE**: *poetry run train --help*   
6.1.(optional) Register your script in pyproject.toml. This way you can run it without specifying a full path to a script file. (2 points) - **DONE**: *poetry run train --help* to see parameters
7. Choose some metrics to validate your model (at least 3) and calculate them after training. Use K-fold cross-validation. (10 points maximum: 2 per metric + 4 for K-fold. Note for reviewers: K-fold CV may be overwritten by nested CV if the 9th task is implemented, check the history of commits in this case. If more than 3 metrics were chosen, only 3 are graded) - **DONE**: set *-USV* or *--use_cross_validate* parameter to use cross-validation; use parameter *-NS* or *--n_splits* to set number of splits
8. Conduct experiments with your model. Track each experiment into MLFlow. Make a screenshot of the results in the MLFlow UI and include it in README. You can see the screenshot example below, but in your case, it may be more complex than that. Choose the best configuration with respect to a single metric (most important of all metrics you calculate, according to your opinion). **DONE**  
8.1. Try at least three different sets of hyperparameters for each model. (3 points) **DONE**  
8.2. Try at least two different feature engineering techniques for each model. (4 points) **DONE**  
8.3. Try at least two different ML models. (4 points) **DONE**  
![scr3](https://user-images.githubusercontent.com/62016699/167383688-41dc21c4-4368-4fe1-876e-3674ff35db7d.PNG)
9. Instead of tuning hyperparameters manually, use automatic hyperparameter search for each model (choose a single metric again). Estimate quality with nested cross-validation, e.g. as described here. Although you used a single metric for model selection, the quality should be measured with all the metrics you chose in task 7. (10 points) **DONE**
![scr4](https://user-images.githubusercontent.com/62016699/167461113-861c3858-2f62-4e4b-ba51-3bc2576ce7de.PNG)
10. In your README, write instructions on how to run your code (training script and optionally other scripts you created, such as EDA). If someone who cloned your repository correctly follows the steps you describe, the script should work for them and produce the same results as it produced on your PC (so don't forget about specifying random seeds). The instructions should be as unambiguous and easy to follow as possible. (10 points) **DONE - SEE BELOW**  
10.1. (optional) If you do the optional tasks below, add a development guide to README. You should specify what other developers should do to continue working on your code: what dependencies they should install, how they should run tests, formatting, etc. (2 points) **DONE - SEE BELOW**  
11. (optional) Test your code. Tests should be reproducible and depend only on your code, not on the data folder or external resources. To provide some data for a test, you should either generate random data during the test run or put a small sample of real data to the tests/ folder, that will be excluded from gitignore. You should have at least one test that describes a situation when everything works fine (exit code 0, valid model produced), and one test that checks if it fails or returns an error message with invalid usage. Since you're working with files, you'll also need to create an isolated filesystem that will temporarily store test files and remove them after tests are finished. Read how to do it here for click, or find the solution yourself if you use other options for CLI. Provide a screenshot that tests are passed.  
11.1 (optional) Single or more tests for error cases without using fake/sample data and filesystem isolation, as in the demo. (3 points) **DONE** - *pytest tests/test_origin_data.py*  
11.2 (optional) Test for a valid input case with test data, filesystem isolation, and checking saved model for correctness. (5 points) **DONE** - - *pytest tests/test_with_fake_data.py*
12. (optional) Format your code with black and lint it with flake8. Provide a screenshot that linting and formatting are passed. (2 points) **DONE**
![scr5](https://user-images.githubusercontent.com/62016699/167615079-52b80b47-cef6-4120-a8d2-32e5efa1b650.PNG)
13. (optional) Type annotate your code, run mypy to ensure the types are correct. It's not necessary to use strict mode as in the demo, but make sure all of the methods you implemented are type annotated and used correctly throughout the code. Provide a screenshot of mypy report, it should be successful. (3 points) **DONE**
![scr6](https://user-images.githubusercontent.com/62016699/167648669-408c2554-f775-4253-b913-b80443f085d4.PNG)

  
## Short user guide
This package allows you to train model for classifing the cover type of trees.
1. Clone this repository to your machine.
2. Download [forest](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. View EDA report in your browser
```sh
poetry run eda
``` 
6. Run train with the following command:
```sh
poetry run train -CP <path to csv with data>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
*Short description see below*  

7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
### CLI options description:
Options:

| Short   | Full                | Type        | Default value | Describe                                                      |   
| :---:   | :---:               | :---:       |:---:          |:---:                                                          |
| **-RS** | **--random_state**  | Integer     |42             |Main random_state wherever it is used |
| **-CER** | **--create_eda_report**  | Boolean     |False    |Create and save dataset eda report to data/eda.html  |
| **-SM** | **--save_model**  | Boolean     |True    |Save trained model to file (path set by parameter --output_file_path)  |
| **-OFP** | **--output_file_path**  | File     |data/model.joblib    |Path for saving trained model (switch in parameter --save_model) |
| **-CP** | **--csv_path**  | File     |data/train.csv    |Path to source train csv file  |
| **-T** | **--target**  | Text     |Cover_Type    |Name of target column |
| **-TSR** | **--test_split_ratio**  | 0<Float<1     |0.2    |Test size ratio for splitting dataset, when train whithout K-fold cross-validation  0<x<1  |
| **-DN** | **--drop_na**  |  Boolean    |  True  | Drop NA values |
 | **-US** | **--use_scaler**  |    Boolean  | False   | Use StandartScaler before training |
 | **-UP** | **--use_pca**  |  Boolean    | False   | Use PCA decomposition before training |
 | **-PNC** | **--pca_n_components**  | Integer>1     |    2| PCA: Parameter n_components |
 | **-MN** | **--model_name**  |RFC, DTC, KNN|  RFC  | Training model: "RFC"-RandomForesClassifier,  "DTC"-DecisionTreeClassifier, "KNN"-KNeighborsClassifier  |
 | **-NE** | **--n_estimators**  |  Integer>0    | 100   | RFC ONL:Y: Parameter n_estimators |
 | **-C** | **--criterion**  |  gini or entropy    | gini   |  Parameter criterion for RFC and  DTC |
 | **-MD** | **--max_depth**  |  Integer>0  |  10   |  arameter max_depth for RFC and DTC|
 | **-MF** | **--n_estimators**  |  Integer>=-2    | 0   | Parameter max_features (-2=log2, -1=sqrt, 0=auto,  other - just int values) RFC and DTC|
 | **-SP** | **--splitter**  |    best or random  | best   | DTC ONLY: Parameter splitter |
 | **-BS** | **--bootstrap**  |   Boolean   | True    | RFC ONLY: Parameter bootstrap |
 | **-NN** | **--n_neighbors**  |  Integer>0    |  5 | KNN ONLY: Parameter n_neighbors  |
 | **-W** | **--weights**  |      |  uniform or distance  | KNN ONLY: Parameter weights |
 | **-UCV** | **--use_cross_validate**  |  Boolean    |  True  | Use k-fold cross-validation |
 | **-UAHS** | **--use_automatic_hyperparameter_search**  | Boolean      |  False   | Automatic hyperparameter search using KFold cross-validation |
 | **-NS** | **--n_splits**  |  Integer>1    |  3   | K-fold cross-validation: Parameter n_splits (number of splits) |
 | **-NSO** | **--n_splits_outer**  |  Integer>1   |  5  | K-fold cross-validation: Parameter n_splits (number of splits) for cross_val_score when activated 'Automatic hyperparameter search' |
 | **-NI** | **--n_iter**  |   Integer>1   | 10   | RandomizedSearchCV: parameter n_iter - Number of parameter settings that are sampled |


## Short developer guide
This package allows you to train model for classifing the cover type of trees 
##### Requirements
```sh
[tool.poetry.dependencies]
python = "^3.9"
pandas = "1.3.5"
click = "^8.1.3"
pandas-profiling = "^3.1.0"
sklearn = "^0.0"
scikit-learn = "^1.0.2"
mlflow = "^1.25.1"

[tool.poetry.dev-dependencies]
pytest = "^5.2"
black = "^22.3.0"
mypy = "^0.950"
flake8 = "^4.0.1"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```
1. Clone this repository to your machine.
2. Download [forest](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install
```
5. Select created enviroment. (for VSCode: CTRL+SHIFT+P, then write interpreter and click "Select interpreter", then select enviroment (if not exists - restart VSCode)
6. To check if the code is formatted correctly use  black and flake8 commands (You may replace "src" to "tests")
```sh
black src
```
```sh
flake8 --append-config=flake8.ini src
```
7. For testing model You can run:
for testing with original "forest" dataset 
```sh
pytest tests/test_origin_data.py
```
or with fake temporary dataset:
```sh
pytest tests/test_with_fake_data.py
```  
8. To test code annotation use mypy command (You may replace "src" to "tests"):
```sh
mypy src
```  

