"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Group3.


    Description: This file is used to launch a minimal streamlit web
	application. 
"""
# Streamlit dependencies
import streamlit as st
import pickle
import contractions
from cleaner import handle_weblinks, clean_data, tokenize, transform
from PIL import Image
# Data dependencies
import pandas as pd

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Meta Analytics")
    st.subheader("Climate change tweet classification")
    img = Image.open("resources/imgs/twitter.jpg")
    st.image(img, width = 500,)
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):


            df = pd.DataFrame({'text':[tweet_text]})
            df['text'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
            df['text'] = [' '.join(map(str, l)) for l in df['text']]   
            # handle weblinks, if any
            df['text'] = df['text'].apply(handle_weblinks) 

            # clean the data
            df['text'] = df['text'].apply(clean_data)

            # tokenize the data
            df['text'] = df['text'].apply(tokenize)

            tweet = [" ".join(i) for i in df.text]  


            predictor = pickle.load(open("resources/team3_log_reg.pkl","rb"))


            prediction = predictor.predict(tweet)
            prediction_map = {1:'Positive',2:'Factual',
                             -1:'Negative',0:'Neutral'}
            pred = prediction_map[prediction[0]]

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as a {} Tweet".format(pred))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
