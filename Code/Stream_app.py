import streamlit as st
import AI_Module as am
import Model
import pickle5 as pickle



label_dic1=Model.label_dic1
label_dic2=Model.label_dic2
text = 'Today is an amazing day!'

st.title('산업코드 예측')
user_input = st.text_input("Text", text,key=1)


prob,pred=am.predictor(user_input)

st.write(f'산업코드 예측')
st.write()
st.write()
st.write("------------------")
for i in range(5):    
    st.write(f'{i+1}번째')
    st.write(f'산업코드 : ', label_dic1[pred[i]])
    st.write(f'확률 : ', prob[pred[i]]) 
    st.write("------------------")
    st.write()