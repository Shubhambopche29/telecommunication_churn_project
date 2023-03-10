import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

pickle_in = open("bagging_clf.pkl", "rb")
bagging_clf = pickle.load(pickle_in)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://p4.wallpaperbetter.com/wallpaper/25/1010/360/nature-black-widow-light-digital-wallpaper-preview.jpg");
             background-attachment: fixed;
	     background-position:75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

def welcome():
    return "Welcome All"



def predict_note_authentication(voice_plan,voice_messages,intl_plan,day_min,day_charge,eve_mins,eve_charge,customer_calls):

    prediction =bagging_clf.predict([[voice_plan,voice_messages,intl_plan,day_min,day_charge,eve_mins,eve_charge,customer_calls]])
    return prediction[0]


# noinspection PyUnreachableCode
def main():
    st.title("Customer Churn Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Customer Churn Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    voice_plan = st.text_input("Voice Plan")
    voice_messages = st.text_input("Voice Messages")
    intl_plan = st.text_input("Intl Plan")
    day_min = st.text_input("Day Minutes")
    day_charge = st.text_input("Day Charge")
    eve_mins = st.text_input("Eve Minutes")
    eve_charge = st.text_input("Eve Charge")
    customer_calls = st.text_input("Customer Calls")

    result = ""
    if st.button("Predict"):
        try:
            voice_plan = float(voice_plan)
            voice_messages = float(voice_messages)
            intl_plan = float(intl_plan)
            day_min = float(day_min)
            day_charge = float(day_charge)
            eve_mins = float(eve_mins)
            eve_charge = float(eve_charge)
            customer_calls = float(customer_calls)
            
            result = predict_note_authentication(voice_plan, voice_messages, intl_plan, day_min, day_charge, eve_mins, eve_charge, customer_calls)
            
            st.success('The output is {}'.format(result))
        except:
            st.error("Please input valid numbers for all fields")
            
    if st.button("About"):
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
