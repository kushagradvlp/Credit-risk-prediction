import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('dataset_31_credit-g.csv')

df['checking_status'] = df['checking_status'].str.replace("'", '')
df['credit_history'] = df['credit_history'].str.replace("'", '')
df['purpose'] = df['purpose'].str.replace("'", '')
df['savings_status'] = df['savings_status'].str.replace("'", '')
df['employment'] = df['employment'].str.replace("'", '')
df['personal_status'] = df['personal_status'].str.replace("'", '')
df['property_magnitude'] = df['property_magnitude'].str.replace("'", '')
df['housing'] = df['housing'].str.replace("'", '')
df['job'] = df['job'].str.replace("'", '')

sns.heatmap(df.corr(), annot=True)

df['class'].value_counts()

### Label Encode the target variable

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['class'] = labelencoder.fit_transform(df['class'])

### Create dummies for categorical variables

df=df.join(pd.get_dummies(df['checking_status'],prefix='checking_status'))
df=df.join(pd.get_dummies(df['credit_history'],prefix='credit_history'))
df=df.join(pd.get_dummies(df['purpose'],prefix='purpose'))
df=df.join(pd.get_dummies(df['savings_status'],prefix='savings_status'))
df=df.join(pd.get_dummies(df['employment'],prefix='employment'))
df=df.join(pd.get_dummies(df['installment_commitment'],prefix='installment_commitment'))
df=df.join(pd.get_dummies(df['personal_status'],prefix='personal_status'))
df=df.join(pd.get_dummies(df['other_parties'],prefix='other_parties'))
df=df.join(pd.get_dummies(df['residence_since'],prefix='residence_since'))
df=df.join(pd.get_dummies(df['property_magnitude'],prefix='property_magnitude'))
df=df.join(pd.get_dummies(df['other_payment_plans'],prefix='other_payment_plans'))
df=df.join(pd.get_dummies(df['housing'],prefix='housing'))
df=df.join(pd.get_dummies(df['job'],prefix='job'))
df=df.join(pd.get_dummies(df['num_dependents'],prefix='num_dependents'))
df=df.join(pd.get_dummies(df['own_telephone'],prefix='own_telephone'))
df=df.join(pd.get_dummies(df['foreign_worker'],prefix='foreign_worker'))

df=df.drop(columns=['checking_status',
'credit_history',
'purpose',
'savings_status',
'employment',
'installment_commitment',
'personal_status',
'other_parties',
'residence_since',
'property_magnitude',
'other_payment_plans',
'housing',
'job',
'num_dependents',
'own_telephone',
'foreign_worker'])

independent_var=df[[cols for cols in df.columns if 'class' not in cols]]
dependent_var=df['class']
X_train,X_test,y_train,y_test=train_test_split(independent_var,dependent_var,test_size=0.3,random_state=0)


## Hyperparameter Tuning

model_list=[]
acc_scr=[]
recall_scr=[]
prec_scr=[]
f1_scr=[]
specificity=[]
roc_auc=[]
i=0
dict_classifiers = {
    "Extra Trees": ExtraTreesClassifier(),
}
for model, model_instantiation in dict_classifiers.items():
    print('\n'+model)
    temp=model
    model_list.append(model)
    model = model_instantiation
    print(model_instantiation)
    model=model.fit(X_train,y_train)
    score=model.score(X_train,y_train)
    predictions_model=model.predict(X_test)
    average_precision = average_precision_score(y_test, predictions_model)
    acc_scr.append(accuracy_score(y_test, predictions_model))
    recall_scr.append(recall_score(y_test, predictions_model))
    prec_scr.append(precision_score(y_test, predictions_model))
    f1_scr.append(f1_score(y_test, predictions_model))
    tn, fp, fn, tp =confusion_matrix(y_test, predictions_model).ravel()
    specificity.append(round(float(tn)/(tn+fp)*100,2))
    roc_auc.append(round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])*100,2))
    i=i+1

from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": [i*10 for i in range(1,21)]}
grid = GridSearchCV(estimator=model, param_grid=parameters , cv=None, verbose=True)
grid.fit(independent_var,dependent_var)
print(grid.best_params_)

# Model Trainer

import time

model_list=[]
acc_scr=[]
recall_scr=[]
prec_scr=[]
f1_scr=[]
specificity=[]
roc_auc=[]
time_taken=[]
i=0
dict_classifiers = {
    "Logistic Regression":LogisticRegression(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100),
    "Random Forest": RandomForestClassifier(),
    "Decision Trees": DecisionTreeClassifier(),
    "Gradient Boost": GradientBoostingClassifier(),
    "Naive Bayes":  GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3)
}
for model, model_instantiation in dict_classifiers.items():
    print('\n'+model)
    temp=model
    model_list.append(model)
    start = time.process_time()
    model = model_instantiation
    model=model.fit(X_train,y_train)
    time_taken.append(time.process_time() - start)
    score=model.score(X_train,y_train)
    predictions_model=model.predict(X_test)
    average_precision = average_precision_score(y_test, predictions_model)
    acc_scr.append(accuracy_score(y_test, predictions_model))
    recall_scr.append(recall_score(y_test, predictions_model))
    prec_scr.append(precision_score(y_test, predictions_model))
    f1_scr.append(f1_score(y_test, predictions_model))
    tn, fp, fn, tp =confusion_matrix(y_test, predictions_model).ravel()
    specificity.append(round(float(tn)/(tn+fp)*100,2))
    roc_auc.append(round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])*100,2))
    i=i+1
    plot_confusion_matrix(model,X_test, y_test)
    plt.show()

df_acc=pd.DataFrame(list(zip(model_list,time_taken,acc_scr,recall_scr,prec_scr,f1_scr,specificity,roc_auc)),columns=['Model','Time taken','Accuracy','Recall','Precision','F1 Score','Specificity','AUC'])

df_acc.to_excel('Model_Comparision.xlsx',index=False)


plt.bar(df_acc.Model, df_acc['Time taken'], color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('Time Taken')
plt.title('Comparision of different models\' time for training')
plt.show()

plt.bar(df_acc.Model, df_acc.Accuracy, color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('Accuracy')
plt.title('Comparision of different models\' Accuracy')
plt.show()

plt.bar(df_acc.Model, df_acc.Recall, color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('Recall')
plt.title('Comparision of different models\' Recall')
plt.show()

plt.bar(df_acc.Model, df_acc.Precision, color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('Precision')
plt.title('Comparision of different models\' Precision')
plt.show()

plt.bar(df_acc.Model, df_acc['F1 Score'], color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('F1 Score')
plt.title('Comparision of different models\' F1 Scores')
plt.show()

plt.bar(df_acc.Model, df_acc['Specificity'], color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('Specificity')
plt.title('Comparision of different models\' Specificity')
plt.show()

plt.bar(df_acc.Model, df_acc['AUC'], color=[  'blue', 'green', 'yellow','orange','red'])
plt.xticks(rotation=60)
plt.ylabel('AUC')
plt.title('Comparision of different models\' AUC')
plt.show()
