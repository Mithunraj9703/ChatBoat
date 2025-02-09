import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLP resources
nltk.download('punkt')
nltk.download('stopwords')

# Load a lightweight chatbot model
chatbot = pipeline("text-generation", model="distilgpt2")

# Initialize stopwords
stop_words = set(stopwords.words("english"))

# Function to clean and preprocess user input
def clean_text(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Separate condition handling logic into a different class
class HealthcareResponder:
    def __init__(self, user_input):
        self.user_input = user_input.lower()

    def get_response(self):
        if "symptom" in self.user_input:
            return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
        elif "appointment" in self.user_input:
            return "Would you like me to schedule an appointment with a doctor?"
        elif "medication" in self.user_input:
            return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
        else:
            return None  

# Define chatbot logic
def healthcare_chatbot(user_input):
    cleaned_input = clean_text(user_input)
    
    # Use HealthcareResponder for rule-based responses
    responder = HealthcareResponder(cleaned_input)
    response = responder.get_response()
    
    if response:
        return response
    else:
        try:
            response = chatbot(cleaned_input, max_length=50, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

# Streamlit web app interface
def main():
    st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
    st.title("Healthcare Assistant Chatbot")

    # User input field
    user_input = st.text_input("How can I assist you today?", "")

    # Generate response when the button is clicked
    if st.button("Submit"):
        if user_input.strip():
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.write(f"**User:** {user_input}")
            st.write(f"**Healthcare Assistant:** {response}")
        else:
            st.warning("Please enter a valid query.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
