# UFO-Sightings-pt2: Predicting UFO Sightings Research Outcomes with XGBoost

## Introduction

Greetings, fellow UFO enthusiasts! Today, we're going to embark on an exciting journey through the world of machine learning, using a dataset of UFO sightings to predict research outcomes. We'll be using Python, pandas, and XGBoost, a powerful machine-learning algorithm. So, fasten your seatbelts and prepare for lift-off!

## Prerequisites

Before we begin, make sure you have the following:

- A basic understanding of Python programming.
- Familiarity with pandas, a Python library for data manipulation and analysis.
- Some knowledge of machine learning concepts, particularly classification problems and decision tree-based models.
- Python environment with necessary libraries installed (pandas, numpy, sklearn, xgboost).

## Step 1: Reading the Dataset

Our first step is to read our dataset, which is stored in a CSV file named 'ufo_fullset.csv'. We'll use pandas' `read_csv` function for this.

```python
import pandas as pd

df = pd.read_csv('ufo_fullset.csv')
```

## Step 2: Data Preprocessing

Before we can use our dataset to train a machine learning model, we need to preprocess it. This involves converting categorical variables into numerical ones, handling missing values, and potentially creating new features.

First, let's convert the 'researchOutcome' column into numerical values. This is our target variable, the one we're trying to predict.

```python
df['researchOutcome'] = df['researchOutcome'].map({'unexplained': 0, 'explained': 1, 'probable': 2})
```

Next, we'll remove any rows with missing values:

```python
df = df.dropna()
```

We'll also convert some other categorical variables into numerical ones using one-hot encoding. This involves creating new columns for each unique value in the original column.

```python
df = pd.get_dummies(df, columns=['weather', 'shape'])
```

## Step 3: Training the Model

Now that our data is ready, we can use it to train an XGBoost model. XGBoost stands for "Extreme Gradient Boosting", and it's a decision tree-based algorithm that's known for its speed and performance.

First, we'll split our data into a training set and a testing set:

```python
from sklearn.model_selection import train_test_split

X = df.drop('researchOutcome', axis=1)
y = df['researchOutcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Then, we'll create an XGBoost classifier and train it on our training data:

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)
```

## Step 4: Evaluating the Model

After training the model, we can use it to make predictions on our testing set and evaluate its performance. One common metric for this is accuracy, which is the proportion of correct predictions out of all predictions.

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
```

## Step 5: Hyperparameter Tuning

To further improve our model, we can tune its hyperparameters. This involves trying out different combinations of hyperparameters to find the one that gives the best performance. We'll use GridSearchCV from sklearn for this.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5,

7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200, 500],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 3, 5, 7],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
```

## Conclusion

And there you have it! We've gone through the entire process of using XGBoost to predict research outcomes for UFO sightings, from reading and preprocessing the data to training and evaluating the model. We've also seen how to tune the model's hyperparameters to improve its performance. 

Remember, this is just a starting point. There are many ways to potentially improve the model, such as using different machine learning algorithms, engineering new features, or gathering more data. So keep exploring, keep learning, and most importantly, keep looking up at the sky! You never know what you might see.

Happy coding, and happy UFO hunting!
