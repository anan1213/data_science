#! /usr/env/bin python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from scipy import stats
from scipy.stats import boxcox


def kesson_table(df):
    """this function make kesson_table.
       This can return neccesarry value.
       Returns: kesson_table
    """
    total = df.isnull().sum()
    percent = total / len(df) * 100
    kesson_table = pd.concat([total, percent], keys=['Total', 'Percent'], axis=1)
    kesson_table = kesson_table.drop(kesson_table[kesson_table['Total'] == 0].index)
    return kesson_table

def display_graph(history):
    """this function display two figure.
       two figure are learned figure of deep learning using keras.
       Returns: two figures
    """
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy', fontsize=20)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('accuracy', fontsize=15)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss', fontsize=20)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.show()

def diff_col(df_train, df_test):
    """this function display different columns.
       using end of feature engineerning.
       Returns: diff columns.
    """
    diff_train = list(set(df_train.columns) - set(df_test.columns))
    diff_test = list(set(df_train.columns) - set(df_test.columns))
    diff_all = set(diff_train + diff_test)
    return diff_all


def list_from_json(x, name):
    """this function make column named xx_list.
       (xx_list make by yourself. this func make value.)
       this can' processe in tha case of including NaN.
       before use this function, you must do fillna('none').
       Returns: list 
    """
    if x == 'none':
        return 'none'
    else:
        return_list = []
        jsons = eval(x)
        for json in jsons:
            return_list.append(json[name])
        return return_list


def make_unique_list(df, col_of_list):
    """this function make unique list of having list of column.
       if column of df is not list type, this function don't use.
       and if the col have Nan, you must change 'none'.
       Returns unique_list.
    """
    cols = df[col_of_list]
    unique_list = []
    for col in cols:
        for word in col:
            if word not in unique_list:
                unique_list.append(word)
    del_list = 'none'
    for del_word in del_list:
        try:
            unique_list.remove(del_word)
        except:
            pass
    unique_list.append('none')
    return unique_list


def return_one(x, word):
    """this function returns one-hot list.
       if word exist in x, return 1. else 0.
       Return one-hot.
    """
    if word in x:
        return 1
    else:
        return 0
    

def return_one_list(x, word_list):
    """this function return one-hot list.
       x must be list.
       this func judge existing word_list in x_list. 
       Return one-hot.
    """
    for x_one in x:
        if x_one in word_list:
            return 1
        else:
            return 0
        

def create_getdummies(df, col_name, col_list):
    """this  function make get_dummies. don't use drop_first.
       to use this function, you must make col_list of col_name.
       if column including list don't exist, you can't use this.
       this function use make_unique_list and return_one
       Returns: get_dummies.(original columns)
    """
    unique_list = make_unique_list(df, col_list)
    for unique in unique_list:
        df[col_name + '_' + unique] = pd.Series(np.zeros(df.shape[0]))
        df[col_name + '_' + unique] = df[col_list].apply(lambda x: return_one(x, unique))
        
    
def drop_all(df1, df2, Drop_list):
    """this function drop train test columns.
       this can drop together.
       Drop_list must be list. 
       Return none.
    """
    df1.drop(Drop_list, axis=1, inplace=True)
    df2.drop(Drop_list, axis=1, inplace=True)
    

def fillna_all(df1, df2, col_name, var='none'):
    """this function fillna together df_train ,df_test.
       if arg's str is int, fillna(digits), else fillna('none')
       default is 'none'.
       and your dataframe are must be df_tain, df_test
       Returns: none.
    """
    if var == 'none':
        df1[col_name] = df1[col_name].fillna('none')
        df2[col_name] = df2[col_name].fillna('none')
    else:
        df1[col_name] = df1[col_name].fillna(var)
        df2[col_name] = df2[col_name].fillna(var)


def dict_count(df, col_list, unique_list):
    """this function make dictionry that key is unique-word , value is count.
       col_list must be list.
       if value of one_col exist in unique_list +1.
       for example ['a', 'b', 'c'] all data count.
       And sort value.
       Returns: dictionary 
    """
    dictionary = {}
    for cols in df[col_list]:
        for col in cols:
            if col not in dictionary:
                dictionary[col] = 0
            else:
                pass
            if col in unique_list:
                dictionary[col] += 1
    sort = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return sort

def famous_worker_list(df, col_name, job_name, target):
    """this function make dictionary that have worker power.
       for exalmple , spilbarg -> 20 power so on.
       Returns: dictionary.
    """
    job_dictionary = {}
    name_count = {}
    for i, cols in enumerate(df[col_name]):
        if cols == 'none':
            job_dictionary['none'] = 0
            name_count['none'] = 0
        else:
            for col in cols:
                if col['job'] == job_name:
                    if col['name'] not in job_dictionary:
                        job_dictionary[col['name']] = df[target][i]
                        name_count[col['name']] = 1
                    else:
                        job_dictionary[col['name']] += df[target][i]
                        name_count[col['name']] += 1
                        
    for name_key in job_dictionary:
        job_dictionary[name_key] = job_dictionary[name_key] / (name_count[name_key] + 1)
        
    sort = sorted(job_dictionary.items(), key=lambda x: x[1], reverse=True)
    dict_sort = dict(sort)
    return dict_sort


def famous_list(df, col_name, name, target):
    """this function make dictionary that have worker power.
       for exalmple , spilbarg -> 20 power so on.
       Returns: dictionary.
    """
    dictionary = {}
    count = {}
    for i, cols in enumerate(df[col_name]):
        if cols == 'none':
            dictionary['none'] = 0
            count['none'] = 0
        else:
            for col in cols:
                if col['name'] not in dictionary:
                    dictionary[col['name']] = df[target][i]
                    count[col['name']] = 1
                else:
                    dictionary[col['name']] += df[target][i]
                    count[col['name']] += 1  
                        
    for name_key in dictionary:
        dictionary[name_key] = dictionary[name_key] / (count[name_key] + 1)
        
    sort = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    dict_sort = dict(sort)
    return dict_sort



def open_json(df1, df2, col_name):
    """this function open the json columns.
       before use this function , must fillna('none')
       Returns: new column
    """
    df1['new_'+ col_name] = df1[col_name].apply(lambda x: 'none' if x == 'none' else eval(x))
    df2['new_'+ col_name] = df2[col_name].apply(lambda x: 'none' if x == 'none' else eval(x))

def worker_power(x, dict_power, job_name):
    """this function returns sum worker_power of one column.
       how to use -> df[] = df[].apply(lambda x: worker_power(x, dict_power, job_name))
       Rerturns: sum worker_power.
    """
    count = 0
    if x == 'none':
        return count
    else:
        for list_crew in x:          
            if list_crew['job'] == job_name:
                if list_crew['name'] in dict_power:
                    count += dict_power[list_crew['name']]
            else:
                count += 0
        return count
    
    
def power(x, dict_power, job_name):
    """this function returns sum worker_power of one column.
       how to use -> df[] = df[].apply(lambda x: worker_power(x, dict_power, job_name))
       Rerturns: sum worker_power.
    """
    count = 0
    if x == 'none':
        return count
    else:
        for list_crew in x:          
            if list_crew['name'] in dict_power:
                count += dict_power[list_crew['name']]
            else:
                count += 0
        return count
    
    
def norm_box(value):
    """this function return value of boxcox.
       if value of arg is <= 0, this add abs(value) + 1
       Returns: boxcox
    """
    value_min = value.min()
    if value_min <= 0:
        box = value + np.abs(value_min) + 1
        box, lamda = boxcox(box)
        return box
    else:
        box, lamda = box(value)
        return box
    
    
def all_power(df1, df2, unique_list):
    """this function execution list worker make var.
       Returns: new_columns
    """
    for unique in unique_list:
        power_name = famous_worker_list(df1, 'new_crew', unique, 'log_rev')
        df1[unique + '_var'] = df1['new_crew'].apply(lambda x: worker_power(x, power_name, unique))
        df2[unique + '_var'] = df2['new_crew'].apply(lambda x: worker_power(x, power_name, unique))
        df1[unique + '_var'] = norm_box(df1[unique + '_var'])
        df2[unique + '_var'] = norm_box(df2[unique + '_var'])

        
        
def all_power2(df1, df2, unique_list, col_name):
    """this function execution list worker make var.
       Returns: new_columns
    """
    for unique in unique_list:
        power_name = famous_list(df1, col_name, unique, 'log_rev')
        df1[unique + '_var'] = df1[col_name].apply(lambda x: power(x, power_name, unique))
        df2[unique + '_var'] = df2[col_name].apply(lambda x: power(x, power_name, unique))
        df1[unique + '_var'] = norm_box(df1[unique + '_var'])
        df2[unique + '_var'] = norm_box(df2[unique + '_var'])

        
def plot_boxcox(x):
    """this function display normal value and value of changing by boxcox.
       Returns: two graph.
    """
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(12, 10)
    y = boxcox(x)
    
    stats.probplot(x, dist='norm', fit=True, plot=ax[0])
    stats.probplot(y, dist='norm', fit=True, plot=ax[1])
    
def select_number_limit(sort, num):
    select_list = []
    for select in sort:
        if sort[select] > num:
            select_list.append([select, sort[select]])
    select_list = dict(select_list)
    return select_list

def select_random(sort, num):
    random_index = random.sample(list(sort.items()), num)
    random_index = dict(random_index)
    return random_index