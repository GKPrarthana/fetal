import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('fetal_health.csv')

from sklearn.preprocessing import StandardScaler
data.columns.tolist()

#scaling the features
scale_X = StandardScaler()
X = pd.DataFrame(scale_X.fit_transform(data.drop(['fetal_health'],axis=1),),columns=data.columns.drop(['fetal_health']))
X.head()
X.columns.tolist()

y = data['fetal_health']
y.head()

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train.head()

#logistic regression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train,y_train)

y_pred_lr = log_reg_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_lr)*100
print(f'Accuracy of Logistic Regression Model: {round(accuracy,2)} %')

from sklearn.model_selection import cross_val_score, StratifiedKFold
cv_method = StratifiedKFold(n_splits=3)
#cross validate the LR model
cv_score_lr = cross_val_score(
                            log_reg_model,
                            X_train, y_train,
                            cv = cv_method,
                            n_jobs=2,
                            scoring='accuracy'
                        )
print(f'Scores(cross validate) for Logistic Regression model: \n{cv_score_lr}')
print(f'CrossValMeans: {round(cv_score_lr.mean(),3)*100}%')
print(f'CrossValStandard Deviation: {round(cv_score_lr.std(),3)}')

#hyper parameter tunning LR
from sklearn.model_selection import GridSearchCV
params_lr = {'tol': [0.0001,0.0002,0.0003],
             'C': [0.01,0.1,1,10,100],
             'intercept_scaling': [1,2,3,4]
            }
gridsearchcv_lr = GridSearchCV(estimator=log_reg_model,
                               param_grid=params_lr,
                               cv=cv_method,
                               verbose=1,
                               n_jobs=2,
                               scoring='accuracy',
                               return_train_score=True
                               )
#fit model with train data
gridsearchcv_lr.fit(X_train,y_train)

best_estimator_lr = gridsearchcv_lr.best_estimator_
print(f'Best estimator for LR Model: \n {best_estimator_lr}')

best_params_lr =gridsearchcv_lr.best_params_
print(f'Best parameter values for LR model: {round(gridsearchcv_lr.best_score_, 3)*100}%')

#Based on the results above, after tuning our model(LR), we could boost the model just a little bit. So we keep going with other models.

#check model performance LR
print('Classification Report')
print(classification_report(y_test,y_pred_lr))
#confution matrix LR
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_lr))

ax = plt.subplot()
cf = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cf, annot=True, fmt='0.3g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Normal','Suspect','Pathological']);
plt.show()
#----------------------------------------------------------------
#KNN
knn = KNeighborsClassifier()
knn_mod = knn.fit(X_train,y_train)
print(f'Baseline K-Nearest Neighbours: {round(knn_mod.score(X_test,y_test),3)*100} %')
y_pred_knn = knn_mod.predict(X_test)
#here we are going to tune the baseline model to boost the model
#cross-validate the k-nearest neighbours moddel
cv_method = StratifiedKFold(n_splits=3)
scores_knn = cross_val_score(knn, X_train,y_train,
                             cv=cv_method,
                             n_jobs=2,
                             scoring='accuracy')
print(f'Scores(cross-validate) for K-Nearest Neighbours Model:\n{scores_knn}')
print(f'CrossValMeans: {round(scores_knn.mean(),2)*100}%')
print(f'CrossValStandard Deviation: {round(scores_knn.std(), 3)}')

params_knn = {'leaf_size': list(range(1,30)),
              'n_neighbors': list(range(1,21)),
              'p':[1,2]}

gridsearchcv_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                                param_grid=params_knn,
                                cv=cv_method,
                                verbose=1,
                                n_jobs=1,
                                scoring='accuracy',
                                return_train_score=True
                                )
gridsearchcv_knn.fit(X_train,y_train)

best_estimator_knn = gridsearchcv_knn.best_estimator_
print(f'Best estimator for KNN Model: \n {best_estimator_knn}')

best_params_knn =gridsearchcv_knn.best_params_
print(f'Best parameter values for KNN model: {round(gridsearchcv_knn.best_score_, 3)*100}%')

print('Classification Report')
print(classification_report(y_test,y_pred_knn))

print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_knn))

ax = plt.subplot()
cf = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cf, annot=True, fmt='0.3g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix for KNN');
ax.xaxis.set_ticklabels(['Normal','Suspect','Pathological']);
plt.show()
#-------------------------------------------------------------------------
#random forest
rf = RandomForestClassifier()
rf_mod = rf.fit(X_train,y_train)
print(f'Baseline Random Forest: {round(rf_mod.score(X_test,y_test),3)*100} %')
y_pred_rf = rf_mod.predict(X_test)

cv_method = StratifiedKFold(n_splits=3)
scores_rf = cross_val_score(rf, X_train,y_train,
                             cv=cv_method,
                             n_jobs=2,
                             scoring='accuracy')
print(f'Scores(cross-validate) for RF Model:\n{scores_rf}')
print(f'CrossValMeans: {round(scores_rf.mean(),2)*100}%')
print(f'CrossValStandard Deviation: {round(scores_rf.std(), 3)}')

params_rf = {'min_samples_split': [2,6,20],
             'min_samples_leaf': [1,4,16],
             'n_estimators': [100,200,300,400],
             'criterion': ['gini']
             }

gridsearchcv_rf = GridSearchCV(estimator=RandomForestClassifier(),
                                param_grid=params_rf,
                                cv=cv_method,
                                verbose=1,
                                n_jobs=1,
                                scoring='accuracy',
                                return_train_score=True
                                )
gridsearchcv_rf.fit(X_train,y_train)

best_estimator_rf = gridsearchcv_rf.best_estimator_
print(f'Best estimator for RF Model: \n {best_estimator_rf}')

best_params_rf =gridsearchcv_rf.best_params_
print(f'Best parameter values for RF model: {round(gridsearchcv_rf.best_score_, 3)*100}%')

print('Classification Report')
print(classification_report(y_test,y_pred_rf))

print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_rf))

ax = plt.subplot()
cf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cf, annot=True, fmt='0.3g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix for RF');
ax.xaxis.set_ticklabels(['Normal','Suspect','Pathological']);
plt.show()
#-----------------------------------------------------------------
#gradient boosting classifier
gbc = GradientBoostingClassifier()
gbc_mod = gbc.fit(X_train,y_train)
print(f'Baseline gradient boosting classifier: {round(gbc_mod.score(X_test,y_test),3)*100} %')
y_pred_gbc = gbc_mod.predict(X_test)

cv_method = StratifiedKFold(n_splits=3)
scores_gbc = cross_val_score(gbc, X_train,y_train,
                             cv=cv_method,
                             n_jobs=2,
                             scoring='accuracy')
print(f'Scores(cross-validate) for GB Classifier Model:\n{scores_gbc}')
print(f'CrossValMeans: {round(scores_gbc.mean(),2)*100}%')
print(f'CrossValStandard Deviation: {round(scores_gbc.std(), 3)}')

params_gbc = {'loss': ['log_loss'],
              'learning_rate': [0.05,0.075,0.1,0.25,0.5,0.75,1],
              'n_estimators': [250,500],
              'max_depth': [3,5,8]
            }

gridsearchcv_gbc = GridSearchCV(estimator=GradientBoostingClassifier(),
                                param_grid=params_gbc,
                                cv=cv_method,
                                verbose=1,
                                n_jobs=2,
                                scoring='accuracy',
                                return_train_score=True
                                )
gridsearchcv_gbc.fit(X_train,y_train)

best_estimator_gbc = gridsearchcv_gbc.best_estimator_
print(f'Best estimator for GBC Model: \n {best_estimator_gbc}')

best_params_gbc =gridsearchcv_gbc.best_params_
print(f'Best parameter values for GBC model: {round(gridsearchcv_gbc.best_score_, 3)*100}%')

print('Classification Report')
print(classification_report(y_test,y_pred_gbc))

print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_gbc))

ax = plt.subplot()
cf = confusion_matrix(y_test, y_pred_gbc)
sns.heatmap(cf, annot=True, fmt='0.3g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix for GBC');
ax.xaxis.set_ticklabels(['Normal','Suspect','Pathological']);
plt.show()
#---------------------------------------------------------------------------

#model selection
results = pd.DataFrame({
                        'model': ['Logistic Regression',
                                    'KNN',
                                    'Random Forest',
                                    'Gradint Boosting Classifier'],
                        'score': [log_reg_model.score(X_test,y_test)*100,
                                  knn_mod.score(X_test,y_test)*100,
                                  rf_mod.score(X_test,y_test)*100,
                                  gbc_mod.score(X_test,y_test)*100,
                                  ]
                        })
results_df = results.sort_values(by='score',ascending=False)
results_df = results_df.set_index('model')
results_df.head()


#---------------------------------------------------------------------------
# Predicted values file
y_pred_gbc
test_ids = range(len(y_pred_gbc))

# Create the DataFrame
df_pred = pd.DataFrame({
    'ID': test_ids,
    'FetalHealthProbability': y_pred_gbc
})

# Save to CSV
df_pred.to_csv('predictions.csv', index=False)
df_pred.head(20)

# Plot the distribution of predicted probabilities
df_pred['FetalHealthProbability'].hist(color='skyblue', bins=30, edgecolor='black')
plt.title('Predicted Fetal Health Probability Distribution')
plt.xlabel('Fetal Health Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
