'''Acquire and Prepare telco data from Codeup SQL database'''

########## IMPORTS ##########
import os
import numpy as np
import pandas as pd

import sklearn.preprocessing
from sklearn.model_selection import train_test_split

from env import user, password, host

########## ACQUIRE & PREPARE DATA ##########

def wrangle_telco_data():
    filename = "telco.csv"
    # find if data is cached
    if os.path.isfile(filename):
        # print('csv file found and loaded')
        df = pd.read_csv(filename)
    # cache data if not found
    else:
        # print('creating df and exporting csv')
        # read the SQL query into a dataframe
        df = pd.read_sql(
            '''
            select customer_id, gender, senior_citizen, partner, dependents
                , phone_service, multiple_lines, internet_service_type
                , online_security, online_backup, device_protection, tech_support
                , streaming_tv, streaming_movies
                , contract_type, payment_type, paperless_billing
                , monthly_charges, total_charges, tenure, churn
            from customers 
            join contract_types using(contract_type_id) 
            join internet_service_types using(internet_service_type_id) 
            join payment_types using(payment_type_id)
            ''', f'mysql+pymysql://{user}:{password}@{host}/telco_churn')
        # write that dataframe to disk (caching)
        df.to_csv(filename, index=False)
    # clean and prep
    # where total_charges is ' ', tenure is 0, so total_charges must be 0 when ' '
    df.loc[df.total_charges==' ','total_charges']=0
    # change to float
    df.total_charges = df.total_charges.astype(float)
    # map two-category features to 1/0 for modeling ease
    df['female'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn'] = df.churn.map({'Yes': 1, 'No': 0})
    # create dummy boolean features for categorical columns for modeling ease
    dummy_df = pd.get_dummies(df[['multiple_lines','online_security','online_backup'
                                    ,'device_protection','tech_support','streaming_tv'
                                    ,'streaming_movies','contract_type','internet_service_type'
                                    ,'payment_type']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    # # adding features
    # df['internet_packages'] = df.online_security_Yes + df.online_backup_Yes + df.device_protection_Yes + df.tech_support_Yes + df.streaming_tv_Yes + df.streaming_movies_Yes
    # df['has_internet_package'] = 0
    # df.loc[df.internet_packages>=1,'has_internet_package'] = 1
    # df = df.drop(columns=['internet_packages'])
    # print('data cleaned and prepped')
    # return the dataframe to the calling code
    return df

########## SPLIT DATA ##########

def split_data(df, strat, seed=42, test=.2, validate=.25):
    """
    This function splits a given dataframe into training, validation, and test sets based on a given
    stratification column and specified test and validation sizes.
    
    :param df: The input dataframe that needs to be split into train, validation, and test sets
    :param strat: The strat parameter is the name of the column in the dataframe that will be used for
    stratified sampling. Stratified sampling is a sampling technique where the population is divided
    into subgroups (strata) based on a specific characteristic, and then samples are taken from each
    subgroup to ensure that the sample is
    :param seed: The seed parameter is used to set the random seed for reproducibility. By setting a
    specific seed, the same random split will be generated each time the function is run, defaults to 42
    (optional)
    :param test: The proportion of the data that should be allocated to the test set
    :param validate: The "validate" parameter is the proportion of the data that will be used for
    validation. It is set to 0.25, which means that 25% of the data will be used for validation
    :return: three dataframes: train, validate, and test.
    """
    # print('data split')
    st = [strat]
    train_validate, test = train_test_split(df, test_size=test, random_state=seed, stratify=df[st])
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=seed, 
                                        stratify=train_validate[st])
    # print(f'train -> {train.shape}; {round(len(train)*100/len(df),2)}%')
    # print(f'validate -> {validate.shape}; {round(len(validate)*100/len(df),2)}%')
    # print(f'test -> {test.shape}; {round(len(test)*100/len(df),2)}%')
    return train, validate, test