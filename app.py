import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# load the model from disk
loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
cv_model = pickle.load(open('countvector.pkl', 'rb'))

# text preprocess function
def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return [review]

# Streamlit app
def main():

    # Change theme to dark
    theme = """
        <style>
        .reportview-container {
            background-image: url("images.jpeg");
            background-size: cover;
        }
        </style>
        """
    st.markdown(theme, unsafe_allow_html=True)
    
    st.title("Restaurant Sentiment Analysis")
    st.markdown("Enter your restaurant review and click **Analyze** to get the sentiment analysis result.")

    # Input text
    text = st.text_area("Review", "")

    if st.button("Analyze") and text:
        if text.strip():
            # Perform sentiment analysis
            inp = preprocess(text)
            inp = cv_model.transform(inp).toarray()
            result = loaded_model.predict(inp)[0]

            # Display the sentiment and confidence score
            if result == 1:
                sentiment = "Positive"
                emoji = "😄"
            else:
                sentiment = "Negative"
                emoji = "😞"
            confidence = loaded_model.predict_proba(inp)[0][int(result)]

            st.subheader("Sentiment Analysis Result")
            st.markdown(f"Sentiment: **{sentiment}** {emoji}")
            st.markdown(f"Confidence: **{confidence:.2f}**")
        else:
            st.warning("Please enter a restaurant review.")

if __name__ == "__main__":
    main()

