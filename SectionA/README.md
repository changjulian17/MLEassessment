# Section A
## Question 1 Regression
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook
#### (a) Formulate a hypothesis between x, y, f_xy
Hypothesis that F_xy is invesrsely proportionate to radial length since points near origin had the greatest values. There may be other relationships like 1/(x-y) + C, but was not explored

#### (b) Justify your choice of function and provide best guess for unknown params
In this case we will use OLS using above hypothesis and check R, F score and take the estimate for a and b.

>f_xy ~ 	0.6011/sqrt(x^2+y^2)
>
>
>with a significant R value and p-value < 0.001.

#### (c)(i) Find the PDF of W = Y/X
> The integral of the transformed function is not 1, indicating it is not a valid PDF. This may be due to incorrect estimate of f_xy. Otherwise current function of W still requires normalisation so that CDF of -oo to oo is 1
> However, if the distribution was symmetric and centralised around W = 0, then we can assume 50% CDP on the right then deduct P(0<W<1) to get P(W>1) == P(X<Y).

## Question 2 Ensemble Errors
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook
#### Use Majority Voting to find the expected error rate of the ensemble
> The ensemble error rate is less than either predictor 2 or 3 but more than predictor 1. There is a small chance all 3 predictors are wrong (at 0.9%) but cases where only 2 are wrong cumulates error more than 12%.
#### (b) What can be inferred if the assumption of independence is relaxed on the errors


> In conclusion the accuracy of the model cannot be explained with aggregated accuracy and precision using only the individual predictor error rates. To make the existing model to be applicable, there must be sufficient large sample dataset to represent the population and universe of permutations. Or the predictive capability must be limited a known set of range of dependent variables before error rates and accuracy can be taken at face value. Else the model must consider each individual predictor as independent variables also. The easiest alternative would be to run experiments with dependent variables within the test universe then compute the error of the ensemble instead. This will give the stochastic error within the relevant case scenarios applicable.

## Question 3 Naive Classifier
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook
computed Naive Bayes by hand. Included pseudocount for smoothing.

> Test1: Since `P(X|Y) * P(Y) < P(X|N) * P(N)`,
>Then predict that the first test term will be `N`
> 
> 
> Test2: Since `P(X|Y) * P(Y) < P(X|N) * P(N)`,
> Then predict that the second test term will be `N`

## Question 4 Search
[section-a.ipynb](section-a.ipynb) code and explanations in jupyter notebook

Below functions are search functions to match similar strings across a dataset.


get_indices(data: list) -> list:
    """
    Takes a list of tuples and returns a 2D list of indices.
    Each sub-list corresponds to the indices of all rows pointing to the same person.
    Rows point to the same person if any of their column entries are the same.

    Parameters
    ----------
    data : list of tuples
        The input data where each tuple represents a row.

    Returns
    -------
    list
        A 2D list of indices where each sub-list contains indices of rows pointing
        to the same person.


similarity_score(str1: str, str2: str) -> float:
    """
    Calculates the maximum similarity score between two strings.
    Maximum similarity is the longest sequential string from the short
    string that is present in the long string

    Parameters
    ----------
    str1 : str
        First string for comparison.
    str2 : str
        Second string for comparison.

calculate_similarity(data: list, idx: int) -> list:
    """
    Calculate similarity scores for pairs of texts in the specified column.


## Question 5 NN
[src](src) code is in src folder 
Input layer 0
Hidden layers 1, 3, 5, 7
Dropout layers 2, 4, 6, 8
output layer 9
assumed drop out of 0.3

There was no implementation of 
- residuals between 1 and 3
- momentum
- parameterised hidden layers

[section-a.ipynb](section-a.ipynb) run of fit and predict in this notebook with explanations in jupyter notebook

[requirements.txt](requirements.txt): Prerequisite python libraries to run notebook and source code  

### Dataset
Chosen from the internet as a typical example on non-linear datasets for classification. In this case the concentric
circles cannot be classified with a straight line but a circle. Additionally there is some noise leading to overlap 
increasing the difficulty of the generalisation.

### Results
I first created a 2 then 4 layer fully connected layer which plateaued above 80% accuracy. There is some variance,
Given that the results 

train accuracy ~ 85%

test accuracy ~ 79.5%

There is some variance given that 1% difference in accuracy. However it will be better to review with further Cross Validation with K folds. Also include a confusion matrix and ROC to review the performance. Given only 4 layers, I expected there to be higher bias as the model may not have a specific enough model of the population. However when model is used to check the test data accuracy drops signifiantly, indicating high variance and possible overfitting. Adding the dropout layers can help significantly at improving that performance.


> Final accuracy around 83%

code is in `src` folder
- NeuralNetworkClassifier created with BaseMLP as parent class
- get_data used to extract same dataset as above from scikit-learn
README.md in root folder for reference of Section A