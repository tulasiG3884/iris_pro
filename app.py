import streamlit as st
import numpy as np
import pickle

with open("iris_dataset.pkl",'rb') as f:
    model=pickle.load(f)

st.title("Iris Flower prediction")
sepal_length=st.slider("sepal length(cm)",4.0,8.0,5.8)
sepal_width=st.slider("sepal width(cm)",2.0,4.5,3.0)
petal_length=st.slider("petal length(cm)",1.0,7.0,4.3)
petal_width=st.slider("petal width(cm)",0.1,2.5,1.3)

if st.button("prediction"):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(input_data)
    species=['setosa','Versicolor','Virginica']
    st.success(f"Predicted Iris Species: {prediction[0]}")
