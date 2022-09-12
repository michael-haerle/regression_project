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

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    return pd.read_sql("""SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, 
    lotsizesquarefeet, fips, regionidzip, transactiondate, latitude, longitude, fireplacecnt, fullbathcnt 
    FROM properties_2017 JOIN propertylandusetype USING(propertylandusetypeid) 
    JOIN predictions_2017 USING(parcelid) WHERE propertylandusedesc LIKE 'Single Family Residential';""", get_connection('zillow'))

def get_zillow_data():
    filename = "zillow.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_zillow_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 


def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#defining function filter 
def filter(x):
    if x > 1899 and x < 1910:
        return 1900
    if x > 1909 and x < 1920:
        return 1910
    if x > 1919 and x < 1930:
        return 1920
    if x > 1929 and x < 1940:
        return 1930
    if x > 1939 and x < 1950:
        return 1940
    if x > 1949 and x < 1960:
        return 1950
    if x > 1959 and x < 1970:
        return 1960
    if x > 1969 and x < 1980:
        return 1970
    if x > 1979 and x < 1990:
        return 1980
    if x > 1989 and x < 2000:
        return 1990
    if x > 1999 and x < 2010:
        return 2000
    if x > 2009:
        return 2010


def fips(x):
    if x == 6037:
        return 'Los Angeles, CA'
    if x == 6059:
        return 'Orange, CA'
    if x == 6111:
        return 'Ventura, CA'

#defining function filter 
def filter_square_feet(x):
    if x > 0 and x < 1100:
        return 'small'
    if x > 1099 and x < 2200:
        return 'medium'
    if x > 2199:
        return 'large'

def prepare_zillow(df):
    # renaming columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'square_feet',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',
                              'regionidzip':'region_zip',
                              'lotsizesquarefeet':'lot_square_feet',
                              'transactiondate':'transaction_date',
                              'fireplacecnt':'fireplace',
                              'fullbathcnt':'full_bathroom'})
    # removing outliers
    df = remove_outliers(df, 1.5, ['square_feet', 'tax_value', 'lot_square_feet'])
    # replacing nulls with 0
    df['fireplace'] = df['fireplace'].fillna(0)
    df['full_bathroom'] = df['full_bathroom'].fillna(0)
    # dropped the rows with null values
    df = df.dropna()
    # manualy removing more outliers
    df.drop(df[df.bedrooms == 0].index, inplace=True)
    df.drop(df[df.bathrooms < 1].index, inplace=True)
    df.drop(df[df.bedrooms > 10].index, inplace=True)
    df.drop(df[df.bathrooms > 10].index, inplace=True)
    df.drop(df[df.year_built < 1900].index, inplace=True)
    #applying the filter function to 'year_built' column 
    df['decade'] = df['year_built'].apply(filter)
    # converting lat and lon to correct figures
    df['latitude'] = df.latitude/1000000
    df['longitude'] = df.longitude/1000000
    # applying fips function to 'fips' column
    df['fips_str'] = df['fips'].apply(fips)
    # applying filter_square_feet function to 'square_feet' column
    df['house_size'] = df['square_feet'].apply(filter_square_feet)
    # creating dummy variables
    dummy_df = pd.get_dummies(df[['house_size']], dummy_na=False)
    df = pd.concat([df, dummy_df], axis=1)
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test 


def scale_data(train, 
               validate, 
               test, 
               cols = ['bedrooms', 'bathrooms', 'square_feet', 'lot_square_feet', 'fireplace', 'full_bathroom']):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    #     fit the thing
    scaler.fit(train[cols])
    # applying the scaler:
    train_scaled[cols] = pd.DataFrame(scaler.transform(train[cols]),
                                                  columns=train[cols].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[cols] = pd.DataFrame(scaler.transform(validate[cols]),
                                                  columns=validate[cols].columns.values).set_index([validate.index.values])
    
    test_scaled[cols] = pd.DataFrame(scaler.transform(test[cols]),
                                                 columns=test[cols].columns.values).set_index([test.index.values])
    return train_scaled, validate_scaled, test_scaled


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test


# couldn't get this to work
def imports():
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats
    import pandas as pd
    from math import sqrt

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    import sklearn.preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
    from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
    from sklearn.feature_selection import f_regression 
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_regression, RFE

    import os
    import env