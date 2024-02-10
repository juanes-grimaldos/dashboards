import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
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

st.subheader("Train Set Score: {}".format(round(train_score, 3)))
st.subheader("Test Set Score: {}".format(round(test_score, 3)))
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(['False Negative', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
ax.set_xlabel('Confusion matrix')
st.pyplot(fig)

name = st.text_input("Name of Passenger ")
sex = st.selectbox("Sex", options=['Male', 'Female'])
age = st.slider("Age", 1, 100, 1)
p_class = st.selectbox("Passenger Class", options=['First Class', 'Second Class', 'Third Class'])

sex = 0 if sex == 'Male' else 1
f_class, s_class, t_class = 0, 0, 0
if p_class == 'First Class':
    f_class = 1
elif p_class == 'Second Class':
    s_class = 1
else:
    t_class = 1

predict_dict = LogisticSurvivalModel().get_prediction(sex, age, f_class, s_class, t_class, name)

prediction = predict_dict['prediction']
predict_probability = predict_dict['predict_probability']

if name != '':
    if prediction[0] == 1:
        st.subheader('Passenger {} would have survived with a probability of {}%'.format(name, round(
            predict_probability[0][1] * 100, 3)))
    else:
        st.subheader('Passenger {} would not have survived with a probability of {}%'.format(name, round(
            predict_probability[0][0] * 100, 3)))
