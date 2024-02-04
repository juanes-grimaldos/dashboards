import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model.logistic_survival_model import LogisticSurvivalModel

model_results = LogisticSurvivalModel().clean_df()

train_score = model_results['scores_and_predictions']['train_score']
test_score = model_results['scores_and_predictions']['test_score']

FN = model_results['confusion_matrix']['FN']
TN = model_results['confusion_matrix']['TN']
TP = model_results['confusion_matrix']['TP']
FP = model_results['confusion_matrix']['FP']

st.title("Would you have survived the Titanic Disaster?")
st.subheader("This model will predict if a passenger would survive the Titanic Disaster or not")

st.subheader("Train Set Score: {}".format ( round(train_score,3)))
st.subheader("Test Set Score: {}".format(round(test_score,3)))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
ax.set_xlabel('Confusion matrix')
st.pyplot(fig)
