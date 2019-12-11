'''Functions related to the Ames, Iowa dataset for the Kaggle competition'''
import os
import pandas as pd

#this file path can be changed depending on where you want to work.
#for me, this folder is the root folder for all my ml_learning projects
ROOT_DIR = os.path.join('C:\\', 'users', 'sebas', 'onedrive', 'python', 'machine_learning', 'ml_learning')

#this data path is specific for the iowa housing project, and its where the datasets are stored on my local machine
DATA_PATH = os.path.join(ROOT_DIR, 'datasets', 'iowa_housing')


def load_iowa(filename, data_path=DATA_PATH):
    '''This function will load the data as a pandas dataframe. it takes the filename 
    as an argument which should be the name that the csv file is saved as in your directory.'''
    
    csv_path = os.path.join(data_path, filename)
    
    iowa_data = pd.read_csv(csv_path).fillna(0) #we import the data with the null values replaced with 0s right off the bat. this avoids us having to do it later
    iowa_data.set_index('Id', inplace=True)
    iowa_data = iowa_data.apply(lambda x: x.astype('|S') if x.dtype == 'object' else x, axis=0)
    
    return iowa_data