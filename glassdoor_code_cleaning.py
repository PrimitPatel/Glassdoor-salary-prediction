# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:57:29 2020

@author: primi
"""

import pandas as pd 
import time


df = pd.read_csv('glassdoor_data.csv')

# remove missing values in salary section
df.dropna(subset=['Salary Estimate'], inplace=True)  # remove missing values in salary section

# distinguise salary based on per hour and employeer provided salary
df['Hourly_salary'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0) 
df['Employer_provided_salary'] = df['Salary Estimate'].apply \
    (lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

# Salary column contains  many negative numbers so remove these rows as salary should be integer
df=df[df['Salary Estimate']!='-1']  

# remove (Glassdoor est.) from salary column
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

# remove K and $ from salary
delete_k_ = salary.apply(lambda x : x.replace('K' ,'').replace('$' ,''))
                            

# remove pr hour and employeer provided salary from salary
delete_hr_eps = delete_k_.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary'] = delete_hr_eps.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = delete_hr_eps.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

#company name should be text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

# get state field from locaton column
df['state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.state.value_counts()
# los angeles is city so get its state CA
df['state'] = df.state.apply(lambda x: 'CA' if x.strip().lower() == 'los angeles' else x.strip())
df.state.value_counts()

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# get age of company fromcurrent year 
year = int(time.strftime("%Y"))
df['age'] = df.Founded.apply(lambda x: x if x <1 else year - x)

# remove first unnamed column 
df = df.drop(['Unnamed: 0'], axis =1)

#parsing of job description 
df['Python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df['SQL'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['Spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['AWS'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['Tableau'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
df['PowerBi'] = df['Job Description'].apply(lambda x: 1 if 'power bi' in x.lower() or 'powerbi' in x.lower() or 'power-bi' in x.lower() else 0)
df['Excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df.columns

df.to_csv('glassdoor_data_cleaned.csv',index = False)