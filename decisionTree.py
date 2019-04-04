
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import re


def get_sirname(Name):
    if re.search(r'\b(mr|master|rev)\b', name, re.I):
        return 1
    elif re.search(r'\b(miss|ms)\b', name, re.I):
        return 2
    elif re.search(r'\b(mrs|jr)\b', name, re.I):
        return 3
    else:
        return 0


train = pd.read_csv('http://bit.ly/kaggletrain')
train


#%%
sex = pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:]
embarked = pd.get_dummies(train.Embarked, prefix='Embarked', )
train = pd.concat([train, sex, embarked], axis=1)
train['age_group'] = train['Age'].apply(
    lambda x: 0 if np.isnan(x) else int(int(x - 1)/10) + 1)
train['sirname'] = train['name'].apply(get_sirname, 1)
train.head()

from sklearn.ensemble import RandomForestClassifier
features = np.array(['Pclass', 'Parch', 'Embarked_C', 'Embarked_Q',
                     'Sex_male', 'age_group', 'sirname'])
clf = RandomForestClassifier()
clf.fit(train[features], train['Survived'])

# from the calculated importances, order them from most to least important
# and make a barplot so we can visualize what is/isn't important
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)
padding = np.arange(len(features)) + 0.5
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, features[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()

# train['income_bins'] = pd.cut(train.monthly_income, bins=15)
# pd.value_counts(df['income_bins'])
# not very helpful

#%%
X = train.loc[:, features]
X.shape
#%%
Y = train.Survived
Y.shape
#%%
Y_pred = clf.predict(X)
print('Correctly predicted on TRAINING SET: {}, errors:{}'.format(
    sum(Y == Y_pred), sum(Y != Y_pred)))
#%%
from sklearn.metrics import classification_report, accuracy_score
#%%
print(classification_report(Y, Y_pred))
print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(Y, Y_pred)))

# In[85]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
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


test = pd.read_csv('https://bit.ly/kaggletest')

sex = pd.get_dummies(test.Sex, prefix='Sex').iloc[:, 1:]
embarked = pd.get_dummies(test.Embarked, prefix='Embarked', )
test = pd.concat([test, sex, embarked], axis=1)
test['age_group'] = test['Age'].apply(
    lambda x: 0 if np.isnan(x) else int(int(x - 1)/10) + 1)
test['sirname'] = test['Name'].apply(get_sirname, 1)
test.head()


# In[89]:
X_new = test.loc[:, features]
X.shape
# In[92]:
X_new.head()

# In[93]:
X_new.shape
new_pred_class = clf.predict(X_new)


pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}
             ).set_index('PassengerId').to_csv('decisionTree.csv')

new_pred_class

# In[95]:
test["Survived"] = new_pred_class

# In[96]:
test.head()

#%%
