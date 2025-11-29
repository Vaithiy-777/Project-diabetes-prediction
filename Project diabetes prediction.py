import pandas as pd


df = pd.read_csv("C:\\Users\\Vaithiyanathan\\OneDrive\\Documents\\Desktop\\diabetes.csv")

print(df.info())
print(df.describe())

import seaborn as sns
import matplotlib as plt

sns.histplot([df["Glucose"],df["BMI"]], bins=50 ,kde=True,alpha=0.8, palette=["green", "blue"])

k=df.corr()
sns.heatmap(k, annot=True, cmap="bwr")


sns.pairplot(df, hue="Outcome")

#splitting dataset into train and test

X= df.drop(columns="Outcome", axis=1)
Y=df["Outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

dia_en=pd.DataFrame()

res=pd.DataFrame()

dia_en["Outcome"]= Y_test

#Data preprocessing

#from imblearn.over_sampling import SMOTE

#smSMOTE (random_state=42) 2 

#X, Y sm.fit_resample(X, Y)
 
#handling imbalancing class

from imblearn.over_sampling import RandomOverSampler

ros=RandomOverSampler (random_state=42)

X, Y = ros.fit_resample(X, Y)

#data scaling

from sklearn.preprocessing import StandardScaler

data_scaler = StandardScaler()

data_rescaled = data_scaler.fit_transform(X)

X = pd.DataFrame (data_rescaled, columns=X.columns)
print(X)
 
#importing modules

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
#logistic regression

from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()


log_r= LogisticRegression()

log_r.probability=False

log_r.fit(x_train,y_train)

log_r_pred=log_r.predict(x_test)


log_r_precision_score= precision_score(y_test, log_r_pred)

log_r_accuracy_score=accuracy_score(y_test, log_r_pred)

log_r_f1_score=f1_score(y_test, log_r_pred)
log_r_recall_score=recall_score(y_test, log_r_pred)

log_r_fpr, log_r_tpr, log_r_thresholds = roc_curve(y_test, log_r_pred)


log_r_roc_auc = auc (log_r_fpr, log_r_tpr)



res ["Log_R"] = pd.DataFrame({"Log_R": [log_r_accuracy_score, log_r_precision_score, log_r_f1_score, log_r_recall_score]})

cm = confusion_matrix(y_test, log_r_pred, labels=log_r.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=log_r.classes_)

disp.plot()
plt.title("Logistic Regression") 
plt.show()

#decison tree classifer

from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train )

dt_pred=dt.predict(x_test)

dt_pred_proba=dt.predict_proba(x_test)

dt_precision_score= precision_score(y_test, dt_pred)

dt_accuracy_score=accuracy_score (y_test, dt_pred)

dt_f1_score=f1_score(y_test, dt_pred)

dt_recall_score=recall_score(y_test, dt_pred)

dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_pred)

dt_roc_auc = auc(dt_fpr, dt_tpr)


res["DT"] = pd.DataFrame({"DT": [dt_accuracy_score, dt_precision_score,dt_f1_score, dt_recall_score, dt_roc_auc]})

cm1 = confusion_matrix(y_test, dt_pred, labels=dt.classes_)

disp = ConfusionMatrixDisplay (confusion_matrix=cm1, display_labels=dt.classes_)

disp.plot()

plt.title("Decision Tree")

plt.show()


# RandomForestClassifier model

from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()


rf= RandomForestClassifier()

rf.fit(x_train,y_train)

rf_pred=rf.predict(x_test)

rf_accuracy_score=accuracy_score (y_test, rf_pred)
rf_precision_score= precision_score(y_test, rf_pred)

rf_f1_score=f1_score(y_test, rf_pred)

rf_recall_score=recall_score(y_test, rf_pred)
rf_fpr, rf_tpr, rf_thresholds = roc_curve (y_test, rf_pred)


rf_roc_auc = auc (rf_fpr, rf_tpr)


res ["RF"] = pd.DataFrame({"RF": [rf_accuracy_score, rf_precision_score, rf_f1_score, rf_recall_score, rf_roc_auc]})



cm = confusion_matrix(y_test, rf_pred, labels=rf.classes_)

disp = ConfusionMatrixDisplay (cm, display_labels=rf.classes_)

disp.plot()
plt.title("Random Forest")

plt.show()

#knn

from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()



knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

knn_pred=knn.predict(x_test)


knn_precision_score= precision_score(y_test, knn_pred)

knn_accuracy_score=accuracy_score(y_test, knn_pred)

knn_f1_score=f1_score(y_test, knn_pred)

knn_recall_score=recall_score(y_test, knn_pred)

knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_pred)

knn_roc_auc = auc(knn_fpr, knn_tpr)


res["KNN"] = pd.DataFrame({"KNN": [knn_accuracy_score,knn_precision_score,knn_f1_score, knn_recall_score, knn_roc_auc]})


cm = confusion_matrix(y_test, knn_pred, labels=knn.classes_)

disp.plot()
plt.title("KNN")
plt.show()


#navie bayes


from sklearn.naive_bayes import GaussianNB

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()


nb=GaussianNB()

nb.fit(x_train,y_train)

nb_pred = nb.predict(x_test)


nb_precision_score= precision_score(y_test, nb_pred)

nb_accuracy_score=accuracy_score(y_test, nb_pred)

nb_f1_score=f1_score(y_test, nb_pred)

nb_recall_score=recall_score(y_test, nb_pred)

nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, nb_pred)

nb_roc_auc = auc(nb_fpr, nb_tpr)


res["NB"] = pd.DataFrame({"NB": [nb_accuracy_score,nb_precision_score,nb_f1_score,nb_recall_score,nb_roc_auc]})


cm = confusion_matrix(y_test, nb_pred, labels=nb.classes_)
disp = ConfusionMatrixDisplay (confusion_matrix=cm, display_labels=nb.classes_)

disp.plot()

plt.title("Naive Bayes")

plt.show()


#GradientBoostingClassifier model


from sklearn.ensemble import GradientBoostingClassifier



x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()



gb= GradientBoostingClassifier()

gb.fit(x_train,y_train)

gb_pred=gb.predict(x_test)


gb_precision_score= precision_score(y_test, gb_pred)

gb_accuracy_score=accuracy_score(y_test, gb_pred)

gb_f1_score=f1_score(y_test, gb_pred)

gb_recall_score=recall_score(y_test, gb_pred)

gb_fpr, gb_tpr, gb_thresholds = roc_curve(y_test, gb_pred)

gb_roc_auc = auc(gb_fpr, gb_tpr)

res["GB"] = pd.DataFrame({"GB": [gb_accuracy_score,gb_precision_score,gb_f1_score,gb_recall_score, gb_roc_auc]})


cm = confusion_matrix(y_test, gb_pred, labels=gb.classes_)

disp = ConfusionMatrixDisplay (confusion_matrix=cm, display_labels=gb.classes_)

disp.plot()

plt.title("Gradient Boosting")

plt.show()

#AdaBoostClassifier


from sklearn.ensemble import AdaBoostClassifier

x_train, x_test, y_train, y_test=X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()

ab= AdaBoostClassifier()

ab.fit(x_train,y_train)

ab_pred=ab.predict(x_test)


ab_precision_score= precision_score(y_test, ab_pred)

ab_accuracy_score=accuracy_score(y_test, ab_pred)

ab_f1_score=f1_score(y_test, ab_pred)

ab_recall_score=recall_score(y_test, ab_pred)

ab_fpr, ab_tpr, ab_thresholds = roc_curve (y_test, ab_pred)

ab_roc_auc = auc (ab_fpr, ab_tpr)

res["AB"] = pd.DataFrame({"AB": [ab_accuracy_score,ab_precision_score, ab_f1_score, ab_recall_score, ab_roc_auc]})

cm = confusion_matrix(y_test, ab_pred, labels=ab.classes_)

disp = ConfusionMatrixDisplay (confusion_matrix=cm, display_labels=ab.classes_)

disp.plot()

plt.title("AdaBoost")
plt.show()

#Parameter to display ROC curve

plt.figure()

#plt.plot(lr_fpr, 1r_tpr, color='darkorange', 1w=2, label='Linear Regression (area = %0.3f)' % lr_roc_auc)

plt.plot(dt_fpr, dt_tpr, color='red', lw=2, label='Decision Tree (area = %0.3f)' % dt_roc_auc)

plt.plot(rf_fpr, rf_tpr, color='green', lw=2, label='Random Forest (area = %0.3f)' % rf_roc_auc)

plt.plot(nb_fpr, nb_tpr, color='blue', lw=2, label='Naive Bayes (area = %0.3f)' % nb_roc_auc)


plt.plot(log_r_fpr, log_r_tpr, color='yellow', lw=2, label='Logistic Regression (area = %0.3f)' % log_r_roc_auc)

plt.plot(gb_fpr, gb_tpr, color='black', lw=2, label='Gradient Boosting (area = %0.3f)' % gb_roc_auc)


plt.plot(knn_fpr, knn_tpr, color='cyan', lw=2, label='KNN (area = %0.3f)' % knn_roc_auc)

plt.plot(ab_fpr, ab_tpr, color='grey', lw=2, label=' AdaBoost (area = %0.3f)' % ab_roc_auc)

plt.plot([0, 1], [0, 1], color='indigo', lw=2, linestyle='--')

plt.legend(loc=0)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve")



#Writing Performance Metrices to File

#res_od= pd.DataFrame({[res.keys]: [res.values]}, index=["accuracy", "precision", "f1", "recall"])

res.to_csv("./project_diabetes_results.csv")


















