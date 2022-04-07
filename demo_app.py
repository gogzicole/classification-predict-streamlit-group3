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
    
    
    img = Image.open("resources/imgs/twitter.jpg")
    img2 = Image.open("resources/imgs/meta_analytics.png")
    img3 = Image.open("resources/imgs/Digital.jpg")
    img4 = Image.open("resources/imgs/sample_recording.gif")
    img5 = Image.open("resources/imgs/twitter2.jpg")
    img6 = Image.open("resources/imgs/twitter3.jpg")
    pos = Image.open("resources/imgs/pos.jpg")
    neg = Image.open("resources/imgs/neg.jpg")
    neu = Image.open("resources/imgs/neu.png")
    fact = Image.open("resources/imgs/fact.jpg")
    
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information","Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.title("Meta Analytics")
        st.image([img2,img3], width = 500,)
        st.subheader("What We Do As a Company")
        st.markdown("""
            We specialize in helping small and large companies build machine learning solutions
            that would help grow their business. Since our inception, we have succesfully carried
            and completed over 5000 projects for high profile clients. Here at Meta Analytics, we 
            offer a wide range of products and services that would suite your every business needs.
            some of our product offerings are:

                - Consultations

                - Investment Advisory

                - Machine Learning

                - Data Analytics

                - Project Management

                - Employee Training

            We also help firms seamlessly transition from on site computing to a cloud based 
            adoption framework.
            Visit out website today at:
        """
        )
        # You can read a markdown file from supporting resources folder
        st.info("www.purplemetaanalytics.co.za/offerings")
        st.image(img4, width = 700,)

    # Building out the predication page
    if selection == "Prediction":
        st.title("Meta Analytics")
        st.subheader("Climate change tweet classification")
        st.image([img5,img6], width = 500,)
        st.info("To Make Predictions, Type in the Tweets Using the Box Below")
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

            # tokenize the data using tokenize function
            df['text'] = df['text'].apply(tokenize)
 

            tweet = [" ".join(i) for i in df.text]  


            predictor = pickle.load(open("resources/team3_log_reg.pkl","rb"))


            prediction = predictor.predict(tweet)
            prediction_map = {1:'Positive',2:'Factual',
                             -1:'Negative',0:'Neutral'}
            pred = prediction_map[prediction[0]]

            # When model has successfully run, will print prediction
            st.success("Text Categorized as a {} Tweet".format(pred))
            if pred == 'Positive':
                st.image(pos, width = 300,)
            elif pred == 'Negative':
                st.image(neg, width = 300,)
            elif pred == 'Neutral':
                st.image(neu, width = 300,)
            elif pred == 'Factual':
                st.image(fact, width = 300,)
            
        
        st.image(img, width = 600,)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
