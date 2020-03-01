import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection as model

col_names = ['Variance of Wavelet Transformed Image',
            'Skewness of Wavelet Transformed Image',
            'Kurtosis of Wavelet Transformed Image',
            'Entropy of Iimage',
            'Class' ]
# load dataset

data_bank = pd.read_csv("data_banknote_authentication.txt", names=col_names)
#print(data_bank)


feature_cols = ['Variance of Wavelet Transformed Image',
            'Skewness of Wavelet Transformed Image',
            'Kurtosis of Wavelet Transformed Image',
            'Entropy of Iimage']

label_col = ['Class']

X = data_bank[feature_cols]
y = data_bank[label_col]

X_train, X_test, y_train, y_test = model.train_test_split( X, y, test_size=0.4, random_state= 27)

#print(X_train)
#print(y_test)
#print(y_train)

logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)
#
y_pred=logreg.predict(X_test)
#print(y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print(metrics.classification_report(y_test,y_pred))
plt.figure()
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
