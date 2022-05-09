Задание к 9-му модулю. 
Датасет https://www.kaggle.com/competitions/forest-cover-type-prediction

Выполнение заданий:
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
![scr3](https://user-images.githubusercontent.com/62016699/167382989-2d5c0405-8518-43a7-9c3d-aabf35121d50.PNG)



