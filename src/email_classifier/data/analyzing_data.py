import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
from email_classifier.config import settings
sns.set()
seperator = '<---------------------->'

data1 = pd.read_csv(RAW_DATA_DIR / "data 1" / "data1.csv")
data2 = pd.read_csv(RAW_DATA_DIR / "data 2" / "data2.csv")
data3 = pd.read_csv(RAW_DATA_DIR / "data 3" / "combined_data.csv")



print(data1.info()) # Shape : (84,4)

print(data2.info()) # Shape : (5172, 3001)

print(data3.info()) # Shape : (83449, 2)

###Conclusion : Data 1 is simply not enough to train a Machine Learning model.
#               Hence, Data 1 (excluded)  

data2.head()
data2['type'].value_counts()
### RESULTS:    
# type
# 0    5015
# 1     124
# 2      15
# 4       8
# 3       6
# 5       1
# 7       1
# 9       1
# 6       1
# Name: count, dtype: int64

### Conclusion : Data2 is also not suitable for this operation
#                As, this does not provie properly un-biased data samples
#                And is useless to use sampling techniques due to the ratio 
#                these types.
#%%


data3.head()
data3['label'].value_counts()
### RESULTS : 
# label
# 1    43910
# 0    39538
# Name: count, dtype: int64         (State : Excellent)

plt.hist(data3['label']) 
### OR ###
sns.countplot(x='label', data=data3, palette='Set2')

### FINAL CONCLUSION : 3rd Dataset is suitable for the current operation.

#######################################  -  #############################################
#######################################  -  #############################################
#######################################  -  #############################################

### ANALYZING DATASET NO. 3

data = data3
data.head()
data.tail()
data.columns    ### Result : ['label', 'text']
data.shape      ### Result : (83448, 2)
data.dtypes     ### Result : label; int64  text; object

data.info()
data.describe()

data.isnull().sum()       ### Result : label; 0   text; 0
data.isnull().mean()*100  ### Result : label; 0.0  text; 0.0

### Since, there is no null values, The need of Report isn't neccessary.

### Code for making Report:
# missing_report = pd.DataFrame({
#     "Missing_Count": data.isnull().sum(),
#     "Missing_Percentage": data.isnull().mean()*100
# })

# missing_report.sort_values(
#     by="Missing_Percentage",
#     ascending=False
# )


data.duplicated().sum()                  ### Result : 0
data.duplicated(subset=["text"]).sum()   ### Result : 2   (!!!)
data[data.duplicated(subset=['text'], keep=False)]

data['text'][12425] ### Result : 'hi' (label: 0)
data['text'][50633] ### Result : 'hi' (label: 1)
data['text'][20516] ### Result : 'unsubscribe' (label: 0)
data['text'][37213] ### Result : 'unsubscribe' (label: 1)

data['label'].value_counts() ### 1:43910  0:39538
data['label'].value_counts(normalize=True)*100 ### Results ==>  1 : 52.6% 
#                                                            ==>  0 : 47.4%
                                               
data['char_count'] = data['text'].str.len()
data['char_count'].describe()

data.groupby('label')['char_count'].describe()


data['text'][52108] ### len(characters) : 598705  (Crazy!!!)

data['word_count'] = data['text'].str.split().str.len()
data.groupby('label')['word_count'].describe()

fig, ax = plt.subplots(1,2, figsize=(8,8))

sns.histplot(data=data['char_count'], bins=50, log_scale=True, ax=ax[0])
ax[0].set_title('Character Count of Emails')

sns.histplot(data=data['word_count'], bins=50, log_scale=True, ax=ax[1])



plt.tight_layout()
plt.show()
sns.boxplot(x=data['char_count'])
sns.violinplot(x=data['char_count'])

data['char_count'][52108] ### Removed Due to Extra Length

# data['text'][52109]
data.to_csv('W:/Machine learning/models/Practice/email-spam-classification/src/data/processed/phase 1.csv')




data['text'].head()
