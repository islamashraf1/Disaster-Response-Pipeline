import sys
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
#%matplotlib inline
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    this function used to load data from Both CSV files and give us joined dataFrame
    input CSV files paths
    output : Dataframe of Joined files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.concat([messages,categories],axis=1)
    return df

def clean_data(df):
    
    '''
    this function used to Clean Data provided 
    input DataFrame
    output : cleaned Data Frame
    '''
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
        # replace numbers of 2 to be 1
        categories[column] = categories[column].str.replace('2','1')
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df,categories],axis=1)
    sum(df.duplicated())
    # drop duplicates
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    '''
    this function used to Save DataFrame to SQL Database
    input DataFrame
    output : SQL Database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('msgs_cat', engine, if_exists = 'replace',index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()