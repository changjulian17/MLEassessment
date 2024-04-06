# Section A
## Question 1 Regression
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook

## Question 2 Ensemble Errors
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook

## Question 3 Naive Classifier
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook

## Question 4 Search
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook

## Question 5 NN
[src](src) code is in src folder 
- base.py
- NeuralNetworkClassifier.py
- get_data
  - gets data from sklearn and split to train and test

[section-a.ipynb](section-a.ipynb) run of fit and predict in this notebook with explanations in jupyter notebook

[requirements.txt](requirements.txt): Prerequisite python libraries to run notebook and source code  

### Dataset
Chosen from the internet as a typical example on non-linear datasets for classification. In this case the concentric
circles cannot be classified with a straight line but a circle. Additionally there is some noise leading to overlap 
increasing the difficulty of the generalisation.

I first created a 2 then 4 layer fully connected layer which plateaued above 80% accuracy. There is some variance,
Given that the results 