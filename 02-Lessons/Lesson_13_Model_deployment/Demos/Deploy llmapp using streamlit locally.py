import streamlit as st
from langchain.llms import Anthropic

import os


# intialize Anthropic model with langchain

os.environ['ANTHROPIC_API_KEY'] = "Enter your own API KEY"
llm = Anthropic(model = "claude-3-5-sonnet-20241022", api_key = os.getenv('ANTHROPIC_API_KEY'), 
               count_tokens = 1024, )



# Streamlit Interface
st.title("Claude with Langchain by Simplilearn")

user_input =  st.text_input("You:", "")

if st.button("Ask Me"):
    if user_input:
        response =  llm.generate(user_input)
        st.write(f"Claude : {response}")
    else:
        st.write("Please enter a question or messsage")









