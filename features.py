#%%
import numpy as np
import pandas as pd
train = pd.read_csv('http://bit.ly/kaggletrain')
train

# In[30]:
# # ONE HOT ENCODING --Maps female to 0 and male 1
sex = pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:]
sex.head()

# In[56]:
# ONE HOT ENCODING --Maps Embarked_C to 1, Embarked_S to 1, and Embarked_Q to 1
embarked = pd.get_dummies(train.Embarked, prefix='Embarked', )
embarked.head()

# %%
import re
X['sirname'] = train['Name'].str.replace(
    r'.*\b(mr|master|miss|ms|mrs|jr)\b.*', r'\1', case=False)
X

# X['sirname'] = train['Name'].str.replace(
#     r'.*\b(mr|master|miss|ms|mrs|jr|rev)\b.*', r'\1', case=False)
# X['age_group'] = train['Age'].apply(
#     lambda x: int(int(x-1)/10)+1 if x == x else None)
# X.dtypes


# def get_sirname(name):
#     if re.search(r'\b(mr|master|rev)\b', name, re.I):
#         return 1
#     elif re.search(r'\b(miss|ms)\b', name, re.I):
#         return 2
#     elif re.search(r'\b(mrs|jr)\b', name, re.I):
#         return 3
#     else:
#         return 0


# X['age_group'] = train['Age'].apply(
#     lambda x: int(int(x-1)/10)+1 if x == x else 0)

# X['sirname'] = train['Name'].apply(get_sirname, 1)

# X.head()

# # In[57]:
# # Concatenates new features back on to orignal dataframe
# train = pd.concat([train, sex, embarked, X.sirname, X.age_group], axis=1)
# train.head()

# # In[58]:
# feature_cols = ['Sex_male', 'Embarked_C', 'Embarked_Q', 'sirname', 'age_group']

# # selecting rows and columns from dataframe ##feature selection essentially
# X = train.loc[:, feature_cols]  # we want every row within feature_cols
# X.shape

# #%%
# # This is the response or TARGET vector
# Y = train.Survived
# Y.shape

# # Now I want to do more feature engineering to add 2 new columns to X.shape so its shape is (891, 5) instead of (891, 3)
# # feature one manipulates Name Series and feature two manipulates Age Series
# # I need your help making using for loop to extract desired info

# # this loop demos the two series I want to mainuplate
# #%%
# #
# for index, row in train.iterrows():
#     print(row["Name"], row["Age"])

# # I'd say for name feature, we classify whether its Mr. Ms. Miss. Mrs. Master
# # For age feature, we classify by every ten years of age with some sort of for loop

# # GOOD LUCK! FUTURE BUSINESS TRANSACTIONS UP FOR GRABS FOR FIRST PPERSON TO HELP SOLVE MY PROBLEMS

# #%%
# # X['sirname'] = train['Name'].str.replace(r'.*\b(mr|master|miss|ms|mrs|jr)\b.*', r'\1', case=False)
# # X

# # X['sirname'] = train['Name'].str.replace(r'.*\b(mr|master|miss|ms|mrs|jr|rev)\b.*', r'\1', case=False)
# # X['age_group'] = train['Age'].apply(lambda x: int(int(x-1)/10)+1 if x == x else None)
# # X.dtypes

# import re


# def get_sirname(name):
#     if re.search(r'\b(mr|master|rev)\b', name, re.I):
#         return 1
#     elif re.search(r'\b(miss|ms)\b', name, re.I):
#         return 2
#     elif re.search(r'\b(mrs|jr)\b', name, re.I):
#         return 3
#     else:
#         return 0


# X['age_group'] = train['Age'].apply(
#     lambda x: int(int(x-1)/10)+1 if x == x else 0)
# X['sirname'] = train['Name'].apply(get_sirname, 1)
# X

# # In[59]:
# from sklearn.linear_model import LogisticRegression  # classification model
# logreg = LogisticRegression()
# logreg.fit(X, Y)

# # In[60]:
# Y_pred = logreg.predict(X)
# print('Correctly predicted on TRAINING SET: {}, errors:{}'.format(
#     sum(Y == Y_pred), sum(Y != Y_pred)))

# # In[61]:
# from sklearn.metrics import classification_report, accuracy_score

# # In[62]:
# print(classification_report(Y, Y_pred))
# print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(Y, Y_pred)))

# # In[85]:
# test = pd.read_csv('https://bit.ly/kaggletest')

# # In[80]:
# test

# # In[86]:
# test['Sex_male'] = test.Sex.map({'female': 0, 'male': 1})

# # In[82]:
# test

# # In[87]:
# embarked = pd.get_dummies(test.Embarked, prefix='Embarked', )
# test = pd.concat([test, embarked], axis=1)

# # In[94]:
# test.head()

# # In[89]:
# X_new = test.loc[:, feature_cols]

# # In[92]:
# X_new.head()

# # In[93]:
# X_new.shape
# new_pred_class = logreg.predict(X_new)


# pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}
#              ).set_index('PassengerId').to_csv('subNew.csv')

# new_pred_class

# # In[95]:
# test["Survived"] = new_pred_class

# # In[96]:
# test.head()

# #%%
