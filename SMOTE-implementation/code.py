# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)

print(df.head(5))

print(df.info)

columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']


for col in columns:
    df[col].replace({'\$':'',',':''},regex=True,inplace=True)


X = df.drop('CLAIM_FLAG',1)

y = df['CLAIM_FLAG'].copy()

count = df['CLAIM_FLAG'].value_counts()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)




# Code ends here


# --------------
# Code starts here

X_train['INCOME'] = X_train['INCOME'].astype('float')
X_train['HOME_VAL'] = X_train['HOME_VAL'].astype('float')
X_train['BLUEBOOK'] = X_train['BLUEBOOK'].astype('float')
X_train['OLDCLAIM'] = X_train['OLDCLAIM'].astype('float')
X_train['CLM_AMT'] = X_train['CLM_AMT'].astype('float')

print('BEFORE',X_train.dtypes)


X_test['INCOME'] = X_test['INCOME'].astype('float')
X_test['HOME_VAL'] = X_test['HOME_VAL'].astype('float')
X_test['BLUEBOOK'] = X_test['BLUEBOOK'].astype('float')
X_test['OLDCLAIM'] = X_test['OLDCLAIM'].astype('float')
X_test['CLM_AMT'] = X_test['CLM_AMT'].astype('float')

print('AFTER',X_train.dtypes)


X_train.isnull()
X_test.isnull()


# Code ends here


# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)


y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].mean(),inplace=True)

X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']].fillna(X_test[['AGE','CAR_AGE','INCOME','HOME_VAL']].mean(),inplace=True)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for cols in columns:
    le = LabelEncoder()
    X_train[cols] = le.fit_transform(X_train[cols].astype(str))
    X_test[cols] = le.transform(X_test[cols].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

# Instantiate logistic regression
model = LogisticRegression(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)




# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)
X_train,y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)

# Code ends here


