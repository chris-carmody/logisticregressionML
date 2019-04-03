
#%%
import numpy as np
import pandas as pd
import re

def get_sirname(name):
    if re.search(r'\b(mr|master|rev)\b', name, re.I):
        return 1
    elif re.search(r'\b(miss|ms)\b', name, re.I):
        return 2
    elif re.search(r'\b(mrs|jr)\b', name, re.I):
        return 3
    else:
        return 0


train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()

# In[30]:
# # ONE HOT ENCODING --Maps female to 0 and male 1
sex = pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:]
sex.head()

# In[56]:
# ONE HOT ENCODING --Maps Embarked_C to 1, Embarked_S to 1, and Embarked_Q to 1
embarked = pd.get_dummies(train.Embarked, prefix='Embarked', )
embarked.head()

# In[57]:
# Concatenates new features back on to orignal dataframe
train = pd.concat([train, sex, embarked], axis=1)
train.head()

# add two more features (age_group and sirname) to train
# for any 'Age' = Nan, set age_group = 0
train['age_group'] = train['Age'].apply(
    lambda x: 0 if np.isnan(x) else int(int(x - 1)/10) + 1)
train['sirname'] = train['Name'].apply(get_sirname, 1)
train.head()

# In[58]:
feature_cols = ['Sex_male', 'Embarked_C',
                'Embarked_Q', 'age_group', 'sirname']

# selecting rows and columns from dataframe ##feature selection essentially
X = train.loc[:, feature_cols]  # we want every row within feature_cols
X.shape

#%%
# This is the response or TARGET vector
Y = train.Survived
Y.shape

# In[59]:
from sklearn.linear_model import LogisticRegression  # classification model
logreg = LogisticRegression()
logreg.fit(X, Y)

# In[60]:
Y_pred = logreg.predict(X)
print('Correctly predicted on TRAINING SET: {}, errors:{}'.format(
    sum(Y == Y_pred), sum(Y != Y_pred)))

# In[61]:
from sklearn.metrics import classification_report, accuracy_score

# In[62]:
print(classification_report(Y, Y_pred))
print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(Y, Y_pred)))

# In[85]:
test = pd.read_csv('https://bit.ly/kaggletest')


def get_sirname(name):
    if re.search(r'\b(mr|master|rev)\b', name, re.I):
        return 1
    elif re.search(r'\b(miss|ms)\b', name, re.I):
        return 2
    elif re.search(r'\b(mrs|jr)\b', name, re.I):
        return 3
    else:
        return 0


test['Sex_male'] = test.Sex.map({'female': 0, 'male': 1})
embarked = pd.get_dummies(test.Embarked, prefix='Embarked', )
test = pd.concat([test, embarked], axis=1)
test.head()
test['age_group'] = test['Age'].apply(
    lambda x: 0 if np.isnan(x) else int(int(x - 1)/10) + 1)
test['sirname'] = test['Name'].apply(get_sirname, 1)
test.head()

# In[89]:
X_new = test.loc[:, feature_cols]
X.shape
# In[92]:
X_new.head()

# In[93]:
X_new.shape
new_pred_class = logreg.predict(X_new)


pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}
             ).set_index('PassengerId').to_csv('logreg.csv')

new_pred_class

# In[95]:
test["Survived"] = new_pred_class

# In[96]:
test.head()

#%%
