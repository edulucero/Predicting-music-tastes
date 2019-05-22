#Import nessesary libraries.
from math import *
import numpy as np
import pandas as pd
from pathlib import Path


# #### Importing First Dataset 'Users'

# In[2]:


#Datasets directory location from one level up. 
dataset_dir = ("/Users/eduardolucero/Documents/AUTUMN 2018/Programming for Data Analysis/Project/Data/")

#Import user.csv into a DataFrame
users_columns = ('User', 'gender', 'age', 'working', 'region', 'music', 'list_own', 'list_back', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19')
input_file = Path(dataset_dir, 'users.csv')
users_raw = pd.read_csv(input_file, sep = ',',header=0,names=users_columns)

# Create a copy of the raw data to clean and use for data exploration 
users = users_raw.copy(deep=True).dropna()


# ##### Cleaning Dataset

# In[3]:


# Create a dictionary object with the values of the list_own and list_back columns and the respecting integer value.

mapping_dict = {
    "1 hour":1,
    "2 hours":2,
    "3 hours":3,
    "4 hours":4,
    "5 hours":5,
    "6 hours":6,
    "7 hours":7,
    "8 hours":8,
    "9 hours":9,
    "10 hours":10,
    "11 hours": 11,
    "12 hours": 12,
    "13 hours": 13,
    "14 hours": 14,
    "15 hours": 15,
    "16 hours": 16,
    "Less than an hour": 0.5, 
    "1":1,
    "2":2,
    "0 Hours":0,
    "0": 0,
    "3": 3,
    "4": 4,
    "16+ hours": 17,
    "5": 5,
    "6": 6,
    "8": 8,
    "10": 10,
    "7": 7,
    "12": 12,
    "9": 9,
    "15": 15,
    "More than 16 hours":17,
    "14": 14,
    "20": 20,
    "16": 16,
    "13": 13,
    "11": 11,
    "18": 18,
    "22": 22,
    "17": 17,
    "24": 24
}

# Replace values in the list_own column with the corresponding values in the mapping_dict and assume missing data is == 0
users["list_own"] = users["list_own"].replace(mapping_dict).fillna(0).astype(int)

# Replace values in the list_back column with the corresponding values in the mapping_dict and assume missing data is == 0
users["list_back"] = users.list_back.replace(mapping_dict).fillna(0).astype(int)

# Convert 'age' column into an int datatype from float 

users['age'] = users['age'].astype(int)

#Convert question 1 to 19 to integer values 

users.loc[:, 'q1':'q19'] = users.loc[:, 'q1':'q19'].fillna(0).astype(int)

# Convert gender, working, region and music to categorical datatypes


users.head()


# #### Importing Second Dataset 'Words'

# In[4]:


# import 'words' dataset into a DataFrame
input_file = Path(dataset_dir, 'words.csv')
words_raw = pd.read_csv(input_file, sep = ',', encoding='latin-1')


# Create a copy of the raw data to clean and use for data exploration 
words = words_raw.copy(deep=True)
words['HEARD_OF'] = words['HEARD_OF'].fillna("Never heard of")
words['OWN_ARTIST_MUSIC'] = words['OWN_ARTIST_MUSIC'].fillna("Own none of their music")
words['LIKE_ARTIST'] = words['LIKE_ARTIST'].fillna(0)

#Drop Empty column
words = words.drop('Unnamed: 87', axis = 1)

# Assume missing values == 0 
words.loc[:, 'Uninspired':'Soulful'] = words.loc[:, 'Uninspired':'Soulful'].fillna(0).astype(int)


# ##### Cleaning Dataset

# In[5]:


# Create dictionary for 'HEARD_OF' column
words_heard_of_dict = {
    "Ever heard music by": "Never heard of",
    "Ever heard of": "Never heard of",
    "Listened to recently": "Heard of and listened to music RECENTLY",
    "Head of and listened to music EVER": "Head of and listened to music NEVER"
}

# Create dictionary for 'OWN_ARTIST_MUSIC' column
words_own_artist_music_dict = {
    "DonÕt know": "Don't Know",
    "DonÍt know": "Don't Know",
    "don`t know": "Don't Know"
}

# Apply replace function to corresponding columns
words["HEARD_OF"] = words.HEARD_OF.replace(words_heard_of_dict)
words["OWN_ARTIST_MUSIC"] = words.OWN_ARTIST_MUSIC.replace(words_own_artist_music_dict)

# Prevuew data clean
words.head()


# #### Importing third dataset 'train'

# In[6]:


input_file = Path(dataset_dir, 'train.csv')
train_raw = pd.read_csv(input_file, sep = ',')
train = train_raw.copy(deep=True)
train.head()


# ### Data Merging

# In[7]:


# Merging 'users' and 'train' datasets

users_train = train.merge(users, on = 'User')


# In[8]:


users_train.head()


# In[9]:


# Merging 'train' and 'words'

train_words = train.merge(words, on = ['User', 'Artist'])

train_words.head()


# In[10]:


#Merging 'user' and 'words' datasets

users_words = words.merge(users, on = "User", how='inner' )
users_words.head()


# In[11]:


# Merging users_words dataset with 'train'

data = users_words.merge(train, how='left')

data.head()


# In[12]:


# double check data dimensioanlity
data.shape


# ### Formatting data for SK-Learn
# Sk-Learn only accepts numeric values

# In[13]:


formatted_data = data.copy(deep=True)

formatted_data = formatted_data.dropna()

# Data processing for machine learning algorithm

# Convert gender column into a binary representaion (0 = male, 1 = female)
formatted_data['gender'] = pd.get_dummies(formatted_data['gender'])

# Get binary representation for 'HEARD_OF' AND 'OWN_ARTIST_MUSIC' for formatted_data and 'train_words' datasets
formatted_data['HEARD_OF'] = pd.get_dummies(formatted_data['HEARD_OF'])
formatted_data['OWN_ARTIST_MUSIC'] = pd.get_dummies(formatted_data['OWN_ARTIST_MUSIC'])

train_words['HEARD_OF'] = pd.get_dummies(train_words['HEARD_OF'])
train_words['OWN_ARTIST_MUSIC'] = pd.get_dummies(train_words['OWN_ARTIST_MUSIC'])

# Create replacement dictionary for following columns

music_dict = {
    "Music is important to me but not necessarily more important": 1,
    "Music means a lot to me and is a passion of mine": 2,
    "I like music but it does not feature heavily in my life": 3,
    "Music is important to me but not necessarily more important than other hobbies or interests": 4,
    "Music is no longer as important as it used to be to me": 5,
    "Music has no particular interest for me": 6
    
}

working_dict = {
    "Employed 30+ hours a week":1,
    "Full-time student": 2,
    "Employed 8-29 hours per week": 3, 
    "Retired from full-time employment (30+ hours per week)":4,
    "Full-time housewife / househusband": 5,
    "Self-employed": 6,
    "Temporarily unemployed": 7,
    "Other": 8,
    "Employed part-time less than 8 hours per week": 9,
    "Retired from self-employment": 10,
    "In unpaid employment (e.g. voluntary work)": 11,
    "Part-time student":12,
    "Prefer not to state": 13,

    
}


region_dict = {
    "North": 1, 
    "South": 2, 
    "Midlands": 3,
    "Northern Ireland": 4
}

formatted_data.music = formatted_data.music.replace(music_dict)
formatted_data.region = formatted_data.region.replace(region_dict)
formatted_data.working = formatted_data.working.replace(working_dict)

formatted_data.head()


# In[14]:


# Process 'users_train' dataset

users_train['gender'] = pd.get_dummies(users_train['gender'])
users_train['music'] = users_train.music.replace(music_dict)
users_train['region'] = users_train.region.replace(region_dict)
users_train['working'] = users_train.working.replace(working_dict)


# #### Remove 'Rating' column from formatted_data to create the model training dataset

# In[15]:


train_data = formatted_data.copy(deep = True)

train_data = train_data.drop('Rating', axis = 1)


# In[16]:


users_train_data = users_train.copy(deep = True)

users_train_data = users_train_data.drop('Rating', axis = 1)


# In[17]:


train_words_traindata = train_words.copy(deep = True)
train_words_traindata = train_words_traindata.drop('Rating', axis = 1)


# #### Convert DataFrames to Numpy Arrays for input into for SK-Learn Machine Learning Algorithm

# First Numpy Array for SK-Learn - All variables

# In[18]:


# Create x component
x1 = train_data.head(10000).as_matrix()


# In[19]:


# Create y component
y1 = formatted_data['Rating'].head(10000).as_matrix()


# Second Numpy Array for SK-Learn - Hand Selected Variables

# In[20]:


# Create x component
train_data_2 = train_data[['Artist', 'User', 'music', 'HEARD_OF','OWN_ARTIST_MUSIC', 'LIKE_ARTIST', 'Track', 'q1', 'q3', ]].copy(deep = True)
x2 = train_data_2.head(10000).as_matrix()


# In[21]:


# Create y component 
y2 = formatted_data['Rating'].head(10000).as_matrix()


# Third Numpy Array for SK-Learn - subset of users_train DataFrame

# In[22]:


# Create x component
x3 = users_train_data.head(10000).as_matrix()


# In[23]:


# Create y component
y3 = users_train['Rating'].head(10000).as_matrix()


# Fourth Numpy Array for SK-Learn - Subset of train_words DataFrame

# In[24]:


# Create x component
x4 = train_words_traindata.head(10000).as_matrix()


# In[25]:


# Create y component
y4 = train_words['Rating'].head(10000).as_matrix()


# <hr style="height:5px;border:none;color:#333;background-color:#333;" />
# # 4. Exploratory data analysis
# 
# This is the main part of the project. Include code, plots, and detailed explanation of your analysis of the data. Be sure to include enough detail so that anyone can follow and understand what you are doing.
# 
# __Create as many code, markdown and raw cells as needed__

# In[27]:


#import nessessary libraries 
import matplotlib.pyplot as plt
from matplotlib import cm 


# In[28]:


# View the gender distribution in the data
data.gender.value_counts(normalize=True).apply(lambda x: f'{x*100:.0f}%')


# In[29]:


# Different attitudes towards music by gender

data.groupby(['music'])['gender'].value_counts(normalize=False).plot(kind='barh', stacked=True)


# In[30]:


# Different attitudes towards music by age

#Create bins for splitting up age.
bins = np.arange(10, 100, 10)
#Create labels
labels = pd.cut(data.age, bins)
#Group music column and binned ages
groupedAge = data.groupby(['music', labels])

groupedAge.size().unstack().plot(kind='barh', figsize=(12, 12), title='Attitudes towards music by age group')


# In[31]:


# Reported listening time by age

data.groupby([labels])['list_back'].mean().plot()


# In[32]:


# Comparing means for question 1 for genders.
# q1: I enjoy actively searching for and discovering music that I have never heard before. `
data.groupby(['gender'])['q1'].mean().plot(kind='barh')


# In[33]:


# Comparing means for question 1 for ages.
# q1: I enjoy actively searching for and discovering music that I have never heard before.
data.groupby(['age'])['q1'].mean().plot()


# In[34]:


# Compaaring means for question 3 for ages.
# q3: I am constantly interested in and looking for more music
data.groupby(['age'])['q3'].mean().plot()


# In[35]:


# Comparing the means for question 6 for age.
# q6: I am not willing to pay for music
data.groupby(['age'])['q6'].mean().plot()


# # SVM Regression

# In[36]:


# Import nessessary SK-Learn components
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# #### First Prediction - All Data

# In[37]:


# Split the data for training and for testing 
x1_train, x1_test, y1_train, y1_test= train_test_split(x1, y1, test_size=0.2, random_state=4)


# In[38]:


# Fit the model
model = svm.SVC()
model.fit(x1_train, y1_train)


# In[39]:


# Return accuracy score
prediction = model.predict(x1_test)

print(accuracy_score(y1_test, prediction))


# #### Second Prediction  - Hand Selected Variables

# In[40]:


# Split the data for training and for testing 
x2_train, x2_test, y2_train, y2_test= train_test_split(x2, y2, test_size=0.2, random_state=4)


# In[41]:


# Fit the model
model = svm.SVC()
model.fit(x2_train, y2_train)


# In[42]:


# Return accuracy score
prediction = model.predict(x2_test)
print(accuracy_score(y2_test, prediction))


# #### Third Prediction - 'users_train' dataset

# In[43]:


# Split the data for training and for testing 
x3_train, x3_test, y3_train, y3_test= train_test_split(x3, y3, test_size=0.1, random_state = 4)


# In[44]:


# Fit the model
model = svm.SVC()
model.fit(x3_train, y3_train)


# In[45]:


# Return accuracy score
prediction = model.predict(x3_test)
print(accuracy_score(y3_test, prediction))


# #### Fourth Prediction 'train_words' dataset

# In[46]:


# Split the data for training and for testing 
x4_train, x4_test, y4_train, y4_test= train_test_split(x4, y4, test_size=0.1, random_state = 4)


# In[47]:


# Fit the model
model = svm.SVC()
model.fit(x4_train, y4_train)


# In[48]:


# Return accuracy score
prediction = model.predict(x4_test)
print(accuracy_score(y4_test, prediction))

