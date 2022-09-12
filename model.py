# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os

# Vis Imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Modeling Imports
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# Custom Module Imports
import wrangle
import env

def select_kbest(X_train, y_train):
    # input for desired features
    n = input("How many desired features? ")
    n = int(n)

    # creating the select k best object and setting the number of features desired
    f_selector = SelectKBest(f_regression, k=n)

    # fitting the data
    f_selector.fit(X_train, y_train)

    # getting a mask of the selected columns
    feature_mask = f_selector.get_support()

    # getting a list of column names
    f_feature = X_train.iloc[:,feature_mask].columns.tolist()

    # print the select k best features
    print(f_feature)

def rfe(X_train, y_train):
    # initialize the ML algorithm
    lm = LinearRegression()
    
    # input for desired features  
    n = input("How many desired features? ")
    n = int(n)

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=n)

    # fit the data using RFE
    rfe.fit(X_train, y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = X_train.iloc[:,feature_mask].columns.tolist()
    
    # prints the ffe features     
    print(rfe_feature)

def corr_heatmap(train):
    # Making a correlation dataframe
    train_corr = train.corr().stack().reset_index(name="correlation")
    # plotting the dataframe
    g = sns.relplot(
        data=train_corr,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="icefire", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8))
    # Setting the labels
    g.set(xlabel="", ylabel="", title='Zillow Correlation Scatterplot Heatmap', aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    # Rotating the x axis 90 degrees to clean it up
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    # Making the legends edge color match the data points
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

def barplot(train):
    plt.figure(figsize = (12, 8))
    ax = sns.barplot(x='decade', y='tax_value', data=train)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    tax_value_avg = train.tax_value.mean()
    plt.axhline(tax_value_avg, label='Tax Value Average')
    plt.legend()
    plt.xlabel('Decade Built')
    plt.ylabel('Tax Value')
    plt.title("Any Decade Above the 1960's is Above the Average Tax Value")
    plt.show()
   
def barplot_chi2(train):
    alpha = 0.05
    null_hyp = 'The Decade Built and Tax Value are independent'
    alt_hyp = 'There is a relationship between tax value and the Decade Built'
    observed = pd.crosstab(train.tax_value, train.decade)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between Tax Value and the Decade Built')   
    print('P-Value', p)
    print('Chi2', round(chi2, 2))
    print('Degrees of Freedom', degf)     

def kdeplot(train):
    LA = train[train.fips == 6037]
    plt.figure(figsize = (12, 8))
    sns.kdeplot(
        data=LA, x="year_built", y="tax_value",
        fill=True, thresh=0, levels=100, cmap="mako")
    plt.title('The Year Built in LA is Indepedent to the Tax Value')
    plt.xlabel('Year Built')
    plt.ylabel('Tax Value')

def kdeplot_chi2(train):
    LA = train[train.fips == 6037]
    alpha = 0.05
    null_hyp = 'The Year Built in LA and Tax value are independent'
    alt_hyp = 'There is a relationship between Tax Value and the Year Built in LA'
    observed = pd.crosstab(LA.tax_value, LA.year_built)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between Tax Value and the Year Built in LA')
    print('P-Value', p)
    print('Chi2', round(chi2, 2))
    print('Degrees of Freedom', degf) 
        
def scatter_mapbox(train):
    fig = px.scatter_mapbox(train, lat="latitude", lon="longitude", hover_name="fips_str", hover_data=["region_zip", "tax_value", "bedrooms", "bathrooms", "square_feet"],
                        color_discrete_sequence=["blue"], color="tax_value", zoom=8, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def baseline_RMSE(y_train, y_validate):
    # 1. Predict tax_value_pred_mean
    tax_value_pred_mean = y_train['tax_value'].mean()
    y_train['tax_value_pred_mean'] = tax_value_pred_mean
    y_validate['tax_value_pred_mean'] = tax_value_pred_mean

    # 2. RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def eval_models(y_train, y_validate, X_train, X_validate, X_test):
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm'] = lm.predict(X_train)

    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lm = y_validate['tax_value_pred_lm'].min()
    replace_lm_avg = y_validate['tax_value_pred_lm'].mean()
    y_validate['tax_value_pred_lm'] = y_validate['tax_value_pred_lm'].replace(replace_lm, replace_lm_avg)

    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lars'] = lars.predict(X_train)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lars = y_validate['tax_value_pred_lars'].min()
    replace_lars_avg = y_validate['tax_value_pred_lars'].mean()
    y_validate['tax_value_pred_lars'] = y_validate['tax_value_pred_lars'].replace(replace_lars, replace_lars_avg)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # Getting rid of the negative predicted value
    replace_lm2 = y_validate['tax_value_pred_lm2'].min()
    replace_lm2_avg = y_validate['tax_value_pred_lm2'].mode()
    y_validate['tax_value_pred_lm2'] = y_validate['tax_value_pred_lm2'].replace(replace_lm2, replace_lm2_avg[0])

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lars), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lm), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lm2), 2))

def poly_test(y_test, X_train, X_test, y_train):
    # We need y_test to be a dataframe 
    y_test = pd.DataFrame(y_test)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_test
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)
        
    # predict test
    y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTest/Out-of-Sample: ", round(rmse_test, 2))
    print("R2 Value:", round(r2_score(y_test.tax_value, y_test.tax_value_pred_lm2), 2))
