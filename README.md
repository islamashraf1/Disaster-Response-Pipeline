# Disaster-Response-Pipeline
ML pipeline applied to Social messages during disaster time to distinguish messages types related to each country departments and normal messages

## Table of Content
1. [Installation](#installation)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Instructions](#Instructions)

## installation
installation files needed numpy, pandas, sqlalchemy, re, NLTK, sqllite3, pickle, Sklearn, plotly and flask libraries.

## Project Motivation
this project Aims to apply ETL Pipeline on this dataset to try to distinguish messages types related to each country departments and normal messages during dusaster time

## File Descriptions
code and comments are divided into 3 main python folder 
1. data folder has disaster_categories.csv, disaster_messages.csv, DisasterResponse.db and process_data.py this file to load Data and clean it then save it to SQL Database
2. Models folder has model file and train_classifier.py file to load data from SQL Database then apply ML pipeline to estimate each message type and save ML model 
3. App folder has HTML templates and run.py file to apply Flask concepts between Webapp and backend of ML pipeline
4. Readme.md file with instrutions to execute this project 

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



