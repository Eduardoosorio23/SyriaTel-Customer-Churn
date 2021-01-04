import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
# sns.set_style('darkgrid')

from imblearn.over_sampling import SMOTE

smote = SMOTE()



def run_model(X, labels, model):

    
    #Spliting the test and train data sets
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state=SEED)
    
    
    
    
    
    #Scaling the data
    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(X_train)
    scaled_data_test = scaler.transform(X_test)

    # Convert into a DataFrame
    scaled_df_train = pd.DataFrame(scaled_data_train, columns=X.columns)
    
    #Adressing call imbalance
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(scaled_data_train, y_train)
    
#     if model == xgb.XGBClassifier():
#         X_train_resampled = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
#         X_test = xgb.DMatrix(X_test, label=y_test)
    
    
    #Creating the classifier
    clf = model

    # Fit the classifier
    clf.fit(X_train_resampled, y_train_resampled)

    # Predict on the test set
    test_preds = clf.predict(scaled_data_test)
    train_preds = clf.predict(X_train_resampled)
    
    
    

    def print_metrics(y_test, test_preds, X_test, y_train, train_preds):
 
        plot_confusion_matrix(clf, X_test, y_test, values_format='.3g')
        
        print('Training Precision: ', precision_score(y_train, train_preds, average='macro'))
        print('Testing Precision: ', precision_score(y_test, test_preds, average='macro'))
        print('Training Recall: ', recall_score(y_train, train_preds, average='macro'))
        print('Testing Recall: ', recall_score(y_test, test_preds, average='macro'))
        print('Training Accuracy: ', accuracy_score(y_train, train_preds))
        print('Testing Accuracy: ', accuracy_score(y_test, test_preds))
        print('Training F1-Score: ', f1_score(y_train, train_preds, average='macro'))
        print('Testing F1-Score: ', f1_score(y_test, test_preds, average='macro'))
        
        
        
        
        
    print_metrics(y_test, test_preds, X_test, y_train_resampled, train_preds)
    
    
   