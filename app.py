import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
import matplotlib.pyplot as plt
st.title('Binary Classification Web App')
st.sidebar.title('Binary Classification Web App')
st.markdown('Are your mushrooms edible or poisonous? 🍄')
st.sidebar.markdown('Are your mushrooms edible or poisonous? 🍄')
def load_data():
    df = pd.read_csv('mushrooms.csv')
    le = LabelEncoder()
    cols = df.columns
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df
data = load_data()
def split_data(df):
    X = df.drop('class', axis=1)
    y = df['class']
    return train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = split_data(data)
if st.sidebar.checkbox('Show raw data',False, key='show_data'):
    st.subheader('Mushroom Dataset')
    st.write(data)
    st.subheader('Training Dataset')
    st.write(x_train, y_train)
    st.subheader('Testing Dataset')
    st.write(x_test,y_test)
    st.subheader('Training Data Size')
    st.write(x_train.shape)
    st.write(y_train.shape)
    st.subheader('Testing Data Size')
    st.write(x_test.shape)
    st.write(y_test.shape)

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader('Confusion Matrix')
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()
    if 'ROC Curve' in metrics_list:
        st.subheader('ROC Curve')
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()   
    if 'Precision Recall Curve' in metrics_list:
        st.subheader('Precision Recall Curve')
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()




class_names = ['edible','poisonous']

st.sidebar.subheader('Classifier')
button = st.sidebar.selectbox('Classifier',('SVM','LR','RF'))
if button == 'SVM':
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('C',0.01,10.0,step=0.01,key='c')
    kernel = st.sidebar.radio('Kernel',('linear','rbf'),key='kernel')
    gamma = st.sidebar.radio('Gamma',('auto','scale'),key='gamma')
    metrics = st.sidebar.multiselect('Choose your metrics',('Confusion Matrix','ROC Curve','Precision Recall Curve'))
    st.subheader('Result obtained from SVC')
    model = SVC(C=c,kernel=kernel,gamma=gamma,probability=True)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    st.write(pred)
    score = accuracy_score(pred,y_test)
    st.subheader('Accuracy Score')
    st.write(score)
    st.subheader('Precision Score')
    st.write(precision_score(y_test,pred))
    st.subheader('Recall Score')
    st.write(recall_score(y_test,pred))
    plot_metrics(metrics)
if button == 'LR':
    st.sidebar.subheader('Model Hyperparameters')
    c = st.sidebar.number_input('C',0.01,10.0,step=0.01,key='c')
    max_iter = st.sidebar.slider('No of iterations',100,500, key = 'max_iter')
    metrics = st.sidebar.multiselect('Choose your metrics',('Confusion Matrix','ROC Curve','Precision Recall Curve'))
    st.subheader('Result obtained from LR')
    model = LogisticRegression(C=c,max_iter=max_iter)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    st.write(pred)
    score = accuracy_score(pred,y_test)
    st.subheader('Accuracy Score')
    st.write(score)
    st.subheader('Precision Score')
    st.write(precision_score(y_test,pred))
    st.subheader('Recall Score')
    st.write(recall_score(y_test,pred))
    plot_metrics(metrics)
if button == 'RF':
    st.sidebar.subheader('Model Hyperparameters')
    n_estimators = st.sidebar.slider('No of trees',1000,5000, key = 'n_estimators')
    metrics = st.sidebar.multiselect('Choose your metrics',('Confusion Matrix','ROC Curve','Precision Recall Curve'))
    st.subheader('Result obtained from RF')
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    st.write(pred)
    score = accuracy_score(pred,y_test)
    st.subheader('Accuracy Score')
    st.write(score)
    st.subheader('Precision Score')
    st.write(precision_score(y_test,pred))
    st.subheader('Recall Score')
    st.write(recall_score(y_test,pred))
    plot_metrics(metrics)
