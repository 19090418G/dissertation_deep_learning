import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time 

# library for sampling
from scipy.stats import uniform

# libraries for Data Download
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Keras
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

class ModelStateReset(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()
reset=ModelStateReset()


def create_shallow_LSTM(epochs=1, 
                        LSTM_units=1,
                        num_samples=1, 
                        look_back=1,
                        num_features=None,  
                        dropout_rate=0,
                        recurrent_dropout=0,
                        verbose=0):
    
    model=Sequential()
    
    model.add(LSTM(units=LSTM_units, 
                   batch_input_shape=(num_samples, look_back, num_features), 
                   stateful=True, 
                   recurrent_dropout=recurrent_dropout)) 
    
    model.add(Dropout(dropout_rate))
            
    model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.he_normal(seed=1)))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


start_sp=datetime.datetime(1980, 1, 1) 
end_sp=datetime.datetime(2019, 2, 28)

yf.pdr_override() 
sp500=pdr.get_data_yahoo('^GSPC', 
                        start_sp,
                        end_sp)
sp500.shape

# Compute the logarithmic returns using the Closing price 
sp500['Log_Ret_1d']=np.log(sp500['Close'] / sp500['Close'].shift(1))

# Compute logarithmic returns using the pandas rolling mean function
sp500['Log_Ret_1w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=5).sum()
sp500['Log_Ret_2w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=10).sum()
sp500['Log_Ret_3w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=15).sum()
sp500['Log_Ret_4w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=20).sum()
sp500['Log_Ret_8w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=40).sum()
sp500['Log_Ret_12w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=60).sum()
sp500['Log_Ret_16w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=80).sum()
sp500['Log_Ret_20w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=100).sum()
sp500['Log_Ret_24w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=120).sum()
sp500['Log_Ret_28w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=140).sum()
sp500['Log_Ret_32w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=160).sum()
sp500['Log_Ret_36w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=180).sum()
sp500['Log_Ret_40w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=200).sum()
sp500['Log_Ret_44w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=220).sum()
sp500['Log_Ret_48w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=240).sum()
sp500['Log_Ret_52w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=260).sum()
sp500['Log_Ret_56w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=280).sum()
sp500['Log_Ret_60w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=300).sum()
sp500['Log_Ret_64w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=320).sum()
sp500['Log_Ret_68w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=340).sum()
sp500['Log_Ret_72w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=360).sum()
sp500['Log_Ret_76w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=380).sum()
sp500['Log_Ret_80w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=400).sum()

# Compute Volatility using the pandas rolling standard deviation function
sp500['Vol_1w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=5).std()*np.sqrt(5)
sp500['Vol_2w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=10).std()*np.sqrt(10)
sp500['Vol_3w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=15).std()*np.sqrt(15)
sp500['Vol_4w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=20).std()*np.sqrt(20)
sp500['Vol_8w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=40).std()*np.sqrt(40)
sp500['Vol_12w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=60).std()*np.sqrt(60)
sp500['Vol_16w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=80).std()*np.sqrt(80)
sp500['Vol_20w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=100).std()*np.sqrt(100)
sp500['Vol_24w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=120).std()*np.sqrt(120)
sp500['Vol_28w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=140).std()*np.sqrt(140)
sp500['Vol_32w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=160).std()*np.sqrt(160)
sp500['Vol_36w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=180).std()*np.sqrt(180)
sp500['Vol_40w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=200).std()*np.sqrt(200)
sp500['Vol_44w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=220).std()*np.sqrt(220)
sp500['Vol_48w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=240).std()*np.sqrt(240)
sp500['Vol_52w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=260).std()*np.sqrt(260)
sp500['Vol_56w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=280).std()*np.sqrt(280)
sp500['Vol_60w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=300).std()*np.sqrt(300)
sp500['Vol_64w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=320).std()*np.sqrt(320)
sp500['Vol_68w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=340).std()*np.sqrt(340)
sp500['Vol_72w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=360).std()*np.sqrt(360)
sp500['Vol_76w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=380).std()*np.sqrt(380)
sp500['Vol_80w']=pd.Series(sp500['Log_Ret_1d']).rolling(window=400).std()*np.sqrt(400)

# Compute Volumes using the pandas rolling mean function
sp500['Volume_1w']=pd.Series(sp500['Volume']).rolling(window=5).mean()
sp500['Volume_2w']=pd.Series(sp500['Volume']).rolling(window=10).mean()
sp500['Volume_3w']=pd.Series(sp500['Volume']).rolling(window=15).mean()
sp500['Volume_4w']=pd.Series(sp500['Volume']).rolling(window=20).mean()
sp500['Volume_8w']=pd.Series(sp500['Volume']).rolling(window=40).mean()
sp500['Volume_12w']=pd.Series(sp500['Volume']).rolling(window=60).mean()
sp500['Volume_16w']=pd.Series(sp500['Volume']).rolling(window=80).mean()
sp500['Volume_20w']=pd.Series(sp500['Volume']).rolling(window=100).mean()
sp500['Volume_24w']=pd.Series(sp500['Volume']).rolling(window=120).mean()
sp500['Volume_28w']=pd.Series(sp500['Volume']).rolling(window=140).mean()
sp500['Volume_32w']=pd.Series(sp500['Volume']).rolling(window=160).mean()
sp500['Volume_36w']=pd.Series(sp500['Volume']).rolling(window=180).mean()
sp500['Volume_40w']=pd.Series(sp500['Volume']).rolling(window=200).mean()
sp500['Volume_44w']=pd.Series(sp500['Volume']).rolling(window=220).mean()
sp500['Volume_48w']=pd.Series(sp500['Volume']).rolling(window=240).mean()
sp500['Volume_52w']=pd.Series(sp500['Volume']).rolling(window=260).mean()
sp500['Volume_56w']=pd.Series(sp500['Volume']).rolling(window=280).mean()
sp500['Volume_60w']=pd.Series(sp500['Volume']).rolling(window=300).mean()
sp500['Volume_64w']=pd.Series(sp500['Volume']).rolling(window=320).mean()
sp500['Volume_68w']=pd.Series(sp500['Volume']).rolling(window=340).mean()
sp500['Volume_72w']=pd.Series(sp500['Volume']).rolling(window=360).mean()
sp500['Volume_76w']=pd.Series(sp500['Volume']).rolling(window=380).mean()
sp500['Volume_80w']=pd.Series(sp500['Volume']).rolling(window=400).mean()

# Label data: Up (Down) if the the 1 month (¡Ö 21 trading days) logarithmic return increased (decreased)
sp500['Return_Label']=pd.Series(sp500['Log_Ret_1d']).shift(-21).rolling(window=21).sum()
sp500['Label']=np.where(sp500['Return_Label'] > 0, 1, 0)

# Drop NA¡äs
sp500=sp500.dropna("index")
sp500=sp500.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', "Return_Label"], axis=1)

# Input arrays should be shaped as (samples or batch, time_steps or look_back, num_features):
X_train_7_lstm=X_train_7.values.reshape(X_train_7.shape[0], 1, X_train_7.shape[1])
X_test_7_lstm=X_test_7.values.reshape(X_test_7.shape[0], 1, X_test_7.shape[1])

# Time Series Split 
dev_size=0.1 
n_splits=int((1//dev_size)-1)   # using // for integer division
tscv=TimeSeriesSplit(n_splits=n_splits) 

steps_b=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
           ('logistic', linear_model.SGDClassifier(loss="log", shuffle=False, early_stopping=False, tol=1e-3, random_state=1))]

pipeline_b=Pipeline(steps_b) # Using a pipeline we glue together the Scaler & the Classifier
# This ensure that during cross validation the Scaler is fitted to only the training folds

# Penalties
penalty_b=['l1', 'l2', 'elasticnet']

# Evaluation Metric
scoring_b={'AUC': 'roc_auc', 'accuracy': make_scorer(accuracy_score)} #multiple evaluation metrics
metric_b='accuracy' #scorer is used to find the best parameters for refitting the estimator at the end

# number of samples
num_samples=1 
# time_steps
look_back=1


# Evaluation Metric
scoring_lstm='accuracy'

start=time.time()


# number of epochs
epochs=1
# number of units 
LSTM_units_7_lstm=220
# numer of features
num_features_7_lstm=X_train_7.shape[1]
# Regularization
dropout_rate=0.
recurrent_dropout=0.4
# print
verbose=0

#hyperparameter
batch_size=[1] 

# hyperparameter
hyperparameter_7_lstm={'batch_size':batch_size}


# create Classifier
clf_7_lstm=KerasClassifier(build_fn=create_shallow_LSTM, 
                           epochs=epochs, 
                           LSTM_units=LSTM_units_7_lstm, 
                           num_samples=num_samples, 
                           look_back=look_back, 
                           num_features=num_features_7_lstm, 
                           dropout_rate=dropout_rate,
                           recurrent_dropout=recurrent_dropout,
                           verbose=verbose)
# Gridsearch
search_7_lstm=GridSearchCV(estimator=clf_7_lstm, 
                           param_grid=hyperparameter_7_lstm,  
                           n_jobs=-1,  
                           cv=tscv, 
                           scoring=scoring_lstm, # accuracy
                           refit=True, 
                           return_train_score=False)

# Fit model
tuned_model_7_lstm=search_7_lstm.fit(X_train_7_lstm, y_train_7, shuffle=False, callbacks=[reset])



print("\n")

# View Accuracy 
print(scoring_lstm +' of the best model: ', tuned_model_7_lstm.best_score_)
# best_score_ Mean cross-validated score of the best_estimator

print("\n")

# View best hyperparameters
print("Best hyperparameters:")
print('epochs:', tuned_model_7_lstm.best_estimator_.get_params()['epochs'])
print('batch_size:', tuned_model_7_lstm.best_estimator_.get_params()['batch_size'])
print('dropout_rate:', tuned_model_7_lstm.best_estimator_.get_params()['dropout_rate'])
print('recurrent_dropout:', tuned_model_7_lstm.best_estimator_.get_params()['recurrent_dropout'])

end=time.time()
print("\n")
print("Running Time:", end - start)

# Make predictions
y_pred_7_lstm=tuned_model_7_lstm.predict(X_test_7_lstm)

# create confustion matrix
fig, ax=plt.subplots()
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test_7, y_pred_7_lstm)), annot=True, cmap="Blues" ,fmt='g')
plt.title('Confusion matrix'); plt.ylabel('Actual label'); plt.xlabel('Predicted label')
ax.xaxis.set_ticklabels(['Down', 'Up']); ax.yaxis.set_ticklabels(['Down', 'Up'])

print("Accuracy:",metrics.accuracy_score(y_test_7, y_pred_7_lstm))
print("Precision:",metrics.precision_score(y_test_7, y_pred_7_lstm))
print("Recall:",metrics.recall_score(y_test_7, y_pred_7_lstm))

y_proba_7_b=tuned_model_7_b.predict_proba(X_test_7)[:, 1]
fpr, tpr, _=metrics.roc_curve(y_test_7,  y_proba_7_b)
auc=metrics.roc_auc_score(y_test_7, y_proba_7_b)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--') # plot no skill
plt.title('ROC-Curve')
plt.show()