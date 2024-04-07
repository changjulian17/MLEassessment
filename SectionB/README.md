# Section B

### Folders
- `data`: given and intermediate data
- `src`: code including models and utils. model is fit and trained then saved before deployment where it is loaded again for /predict endpoint
- `tests`: test scripts
- `Dockerfile`: is used to build image and main.py sets up environment and configures endpoints

http://0.0.0.0:8000/docs
for web app requests


## Assumptions about data
- vectors are all continuous and follow some normal or poison distribution
- titles do not have an influence on target variable
- although it may not be the full song, tracks are long enough to have sufficient data to be classifies

## Data Preprocessing
We will assume that categorical features are only 'time_signature', 'key', 'mode'.

Considered rebalancing Labels since some categories are much more available in the data set than others.
### Data Enrichment¶
Enriching with two more attributes
tags_count, might have some predictive power, some genres will have less words.

longest_tag_length, complexity of words can also be another indicator

Then the string columns are dropped.

## Train Model
SectionB/src/models/train_model.py

Pipeline is created with

- column transformer
  - standard scaler for continuous float variables
  - one hot encoder for cateforical variables
- XGBoost
Model is trained with the pipeline. Cross validated for reference. XGBoost was compared to Scikit-learn MLP for quick reference. Although with more than 2e6 iterations MLP still had very high bias. XGBoost is relatively fast although the variance in accuracy from CV up to 5% difference only.

Model together with pipeline and encoder is saved as pickle file. Then used to predict test.csv and prediction.csv is created.

> Model managed 66% model accuracy on Cross Validated training data. This is higher than only choosing the majority "classic pop and rock" label (21%)


# Webapp
API documentation: http://0.0.0.0:8000/docs#/

### `/predict`
To return a list of genre predictions based on given input. It takes multiple rows of data, but must have all 157 features available. Use XGBoost trained model. Then saving the result using SQLite and pandas

Input format example:

> `{"tracks": [{"trackID":66, ... },{"trackID":55, ... }]}`

Output format example:

> `{"genre": [{66:punk},{55,folk}]}`

### `/genre`
To return a list of song titles from a given genre from music.db. Search function to look up DB.

Input format example:

"punk"

Output format example:

["title1","title2",...]

### Create Database
SQLite and pandas is used to make music.db file in SectionB/data/db folder

It is updated by SectionB/src/utils/dbendpoint.py
