# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:47:00 2020

@author: primi
"""


import glassdoor_data_scraper as gs 
#import pandas as pd 
# set path for chromedriver according
path="D:/Career_learning/glasdoor_job_prediction/glassdor_2/chromedriver"

df = gs.get_jobs('data scientist',300, False, path, 15)

df.to_csv('glassdoor_data.csv', index = False)
df