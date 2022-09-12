# <a name="top"></a>Regression Project
![]()

by: Michael Haerle

<p>
  <a href="https://github.com/michael-haerle" target="_blank">
    <img alt="Michael" src="https://img.shields.io/github/followers/michael-haerle?label=Follow_Michael&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___


## <a name="project_description"></a>Project Description:
Using the data science pipeline to practice with regression. In this repository you will find everything you need to replicate this project. This project uses the Zillow dataset to find key drivers of property value. 

[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
- Create README.md with data dictionary, project and business goals, come up with questions to lead the exploration and the steps to reproduce.
- Acquire data from the Codeup Database and create a function to automate this process. Save the function in an wrangle.py file to import into the Final Report Notebook.
- Clean and prepare data for exploration. Create a function to automate the process, store the function in the wrangle.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- Produce at least 4 clean and easy to understand visuals.
- Clearly define hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Scale the data for modeling.
- Establish a baseline accuracy.
- Train three different classification models.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.


### Project goals: 
- My goal is to find and use key drivers of property value for single family properties sold in 2017 to predict property value with the least amount of error.


### Target variable:
- The target variable for this project is tax value.

### Initial questions:
- Where are the correlations in the data?
- Is there a relationship between Decade Built and Tax Value?
- Is there a relationship between Tax Value and the Year Built in LA?
- What does the data look like on a map?

### Need to haves (Deliverables):
- A final report notebook
- A 5min presentation


### Nice to haves (With more time):
 - If I had more time with the data I would focus on more feature engineering more columns.
 - I would specifically focus more on the square feet of different areas of the property.


### Steps to Reproduce:
- You will need to make an env.py file with a vaild username, hostname and password assigned to the variables user, host, and password
- Then download the wrangle.py, model.py, and final_report.ipynb
- Make sure these are all in the same directory and run the final_report.ipynb.

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- Tax Value has a positive correlation with house_size_large, decade, full_bathroom, year_built, square_feet, bathrooms, and bedrooms.
- Any decade after the 1960's is above the average Tax Value.
- There were only 12 properties sold on Santa Catalina Island.
- Properties near the beach tend to have higher Tax Values.
- In the 1950's there was lots of properties sold, espically those with lower tax values.


***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| bedrooms | Number of bedrooms | float64 |
| bathrooms | Number of bathrooms | float64 |
| square_feet | Square feet of the interior of the house | float64 |
| tax_value | The total tax assessed value of the properity | float64 |
| year_built | Year the house was built | float64 |
| lot_square_feet | Total square feet of the lot | float64 |
| fips |  Federal Information Processing Standard code | float64 |
| region_zip | Zip code | float64 |
| transaction_date | Date the properity was sold | object |
| latitude | Latitude cordinates for the properity | float64 |
| longitude | Longitude cordinates for the properity | float64 |
| fireplace | Number of fireplaces | float64 |
| decade | Decade the house was sold | int64 |
| fips_str | String version of fips used for visuals | float64 |
| house_size | The category of the house size based on the square feet | float64 |
| house_size_large | 1 = House size is in large category, 0 = House size isn't in large category | uint8 |
| house_size_medium | 1 = House size is in medium category, 0 = House size isn't in medium category | uint8 |
| house_size_small | 1 = House size is in small category, 0 = House size isn't in small category | uint8 |
| tax_value_pred_mean | Baseline prediction for tax value using mean | float64 |
| tax_value_pred_median | Baseline prediction for tax value using mean | float64 |
| tax_value_pred_lm | Baseline prediction for tax value using a Linear Regression Model | float64 |
| tax_value_pred_lars | Baseline prediction for tax value using a Lasso + Lars Model | float64 |
| tax_value_pred_glm | Baseline prediction for tax value using a Tweedie Regressor Model | float64 |
| tax_value_pred_lm2 | Baseline prediction for tax value using a Polynomial Features Model | float64 |
***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Prepare steps: 
- Droped duplicate columns
- Created dummy variables
- Concatenated the dummy dataframe
- Renamed columns
- Dropped columns not needed
- Removed ouliers
- Imputed nulls with 0 for fireplace and full_bathroom
- Used square feet to feature engineer a new column where it returned small, medium, or large house size
- Used .apply to apply a custom function to create a decade column for what decade the house was built in
- Converted latitude and longitude to the proper values
- Split into the train, validate, and test sets

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - model.py


### Takeaways from exploration:
- Tax Value has a positive correlation with house_size_large, decade, full_bathroom, year_built, square_feet, bathrooms, and bedrooms.
- Any decade after the 1960's is above the average Tax Value.
- There were only 12 properties sold on Santa Catalina Island.
- Fireplaces does not apear to be useful for the modeling phase.
- Decade and the house_size columns will be used during the modeling phase.

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: The Decade Built and Tax Value are independent.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between tax value and the Decade Built.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- We reject the null hypothesis that The Decade Built and Tax Value are independent
- There is a relationship between tax value and the Decade Built
- 3.9309219442730487e-16
- Chi2 214095.42
- Degrees of Freedom 208846


### Stats Test 2: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: The Year Built in LA and Tax value are independent.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between Tax Value and the Year Built in LA.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- We fail to reject the null hypothesis that The Year Built in LA and Tax value are independent
- There appears to be no relationship between Tax Value and the Year Built in LA
- P-Value 0.4721751128088897
- Chi2 1454990.41
- Degrees of Freedom 1454872


***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline (Using Mean)
    
- Baseline RMSE: 247730.36
    

- Selected features to input into models:
    - features =  ['bedrooms', 'bathrooms', 'square_feet', 'lot_square_feet', 'full_bathroom', 'year_built', 'fips', 'region_zip', 'house_size_large', 'house_size_small', 'decade']

***

## Models:


### Model 1: Random Forest


Model 1 results:
- RandomForestClassifier min_samples_leaf=12, max_depth=8, random_state=123
- Model stats:
- Accuracy: 0.81
- True Positive Rate: 0.51
- False Positive Rate: 0.08
- True Negative Rate: 0.92
- Flase Negative Rate: 0.49
- Precision: 0.71
- Recall: 0.51
- f1 score: 0.59
- Positive support: 1121
- Negative support: 3104
- Accuracy of random forest classifier on training set: 0.81



### Model 2 : K-Nearest Neighbor


Model 2 results:
- KNeighborsClassifier n_neighbors=15
- Model stats:
- Accuracy: 0.81
- True Positive Rate: 0.51
- False Positive Rate: 0.08
- True Negative Rate: 0.92
- Flase Negative Rate: 0.49
- Precision: 0.69
- Recall: 0.51
- f1 score: 0.58
- Positive support: 1121
- Negative support: 3104
- Accuracy of KNN classifier on training set: 0.81

### Model 3 : Logistic Regression

Model 3 results:
- LogisticRegression C=.01, random_state=123, intercept_scaling=1, solver=lbfgs
- Model stats:
- Accuracy: 0.80
- True Positive Rate: 0.46
- False Positive Rate: 0.09
- True Negative Rate: 0.91
- Flase Negative Rate: 0.54
- Precision: 0.66
- Recall: 0.46
- f1 score: 0.55
- Positive support: 1121
- Negative support: 3104
- Accuracy of Logistic Regression classifier on training set: 0.80


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation |
| ---- | ----|
| Baseline | 0.73 |
| Random Forest | 0.80 |
| K-Nearest Neighbor | 0.79 | 
| Logistic Regression | 0.80 |


- {Random Forest} model performed the best


## Testing the Model

- Model Testing Results: 81% Accuracy

***

## <a name="conclusion"></a>Conclusion:

- There are positive correlations with churn and monthly charges, papperless billing, fiber optic, phone service, senior citizen, and those who are above the average monthly charges and below the average tenure.
- There are negative correlations with churn and tenure, total charges, no internet service, and a two year contract.
- There is a relationship between churn and contract type.
- There was not a relationship between both genders who churn and phone service, however there was a relationship between females who churn and phone service.
- There was a relationship between churn and people who are above the average monthly charge and below the average tenure.

#### Idealy we would like to find a way to incentivise more people to sign a one or two year contract as opposed to a month-to-month contract. It would also be benificial to get more people to stay past the average tenure and keep their cost at or below the average monthly charge. 

[[Back to top](#top)]
