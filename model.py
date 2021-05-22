
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle


# In[16]:


loan_train_df = pd.read_csv("train.csv")


# In[17]:


loan_train_df = loan_train_df.drop(columns=['Loan_ID']) ## Dropping Loan ID


# In[18]:


#### Encoding categrical Features: ##########
loan_train_df_encoded = pd.get_dummies(loan_train_df,drop_first=True)


# In[19]:


########## Split Features and Target Varible ############
X = loan_train_df_encoded.drop(columns='Loan_Status_Y')
y = loan_train_df_encoded['Loan_Status_Y']


#my_list = list(X)
#print(my_list)
############### Handling/Imputing Missing values #############

#imp = Imputer(strategy='mean')
#imp_train = imp.fit(X)

#X = imp_train.transform(X)
X=X.fillna(X.mean())

# In[20]:
logreg = LogisticRegression()

#rfc=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 60, max_depth=4, criterion='gini')
logreg.fit(X,y)

# In[21]:


#rfc.fit()


# In[22]:



# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2000, 100, 6000,60,1,0,1,1,0,0,0,0,0,1]]))


