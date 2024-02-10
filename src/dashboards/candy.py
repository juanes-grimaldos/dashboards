import os
os.chdir(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.logistic_model import LogisticGeneralModel

model_results = LogisticGeneralModel('bar').model_architect()

train_score = model_results['train_score']
test_score = model_results['test_score']

st.title("Would it be a candy bar?")
st.subheader("This model will predict the probability of a candy bar being a candy bar")


df = pd.read_csv('data/the-ultimate-halloween-candy-power-ranking/candy-data.csv')
fig = plt.figure(figsize=(10, 5))
df_features = df.drop('competitorname', axis=1)
c = df_features.corr()
sns.heatmap(c, cmap='BrBG', annot=True)
st.pyplot(fig)

st.subheader("Train Set Score: {}".format(round(train_score, 3)))
st.subheader("Test Set Score: {}".format(round(test_score, 3)))

chocolate = st.selectbox("chocolate", options=['yes', 'no'])
fruity = st.selectbox("fruity", options=['yes', 'no'])
caramel = st.selectbox("caramel", options=['yes', 'no'])
peanutyalmondy = st.selectbox("peanutyalmondy", options=['yes', 'no'])
nougat = st.selectbox("nougat", options=['yes', 'no'])
crispedricewafer = st.selectbox("crispedricewafer", options=['yes', 'no'])
hard = st.selectbox("hard", options=['yes', 'no'])
pluribus = st.selectbox("pluribus", options=['yes', 'no'])

chocolate = 1 if chocolate == 'yes' else 0
fruity = 1 if fruity == 'yes' else 0
caramel = 1 if caramel == 'yes' else 0
peanutyalmondy = 1 if peanutyalmondy == 'yes' else 0
nougat = 1 if nougat == 'yes' else 0
crispedricewafer = 1 if crispedricewafer == 'yes' else 0
hard = 1 if hard == 'yes' else 0
pluribus = 1 if pluribus == 'yes' else 0


sugarpercent = st.slider("sugarpercent", 0.1, 1.0, 0.0)
pricepercent = st.slider("pricepercent", 0.1, 1.0, 0.0)
winpercent = st.slider("winpercent", 0.1, 100.0, 0.0)

list_input = [
    chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, pluribus,
    sugarpercent, pricepercent, winpercent
]

predict_dict = LogisticGeneralModel('bar').get_prediction(
    list_input
)

prediction = predict_dict['prediction']
predict_probability = predict_dict['predict_probability']

if prediction[0] == 1:
    st.subheader('This is a candy bar with a probability of {}%'.format(round(
        predict_probability[0][1] * 100, 3)))
else:
    st.subheader('This is not a candy bar with a probability of {}%'.format(round(
        predict_probability[0][0] * 100, 3)))

