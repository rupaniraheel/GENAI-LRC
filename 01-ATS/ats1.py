import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()  # Load environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_repsonse(input):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(input)
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return None

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
        else:
            st.warning(f"Text could not be extracted from page {page}.")
    return text

# Prompt template
input_prompt = """
Hey Act Like a skilled or very experienced ATS(Application Tracking System)
with a deep understanding of the tech field, software engineering, data science, data analytics,
and big data engineering and Digital Marketer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving the resumes. Assign the percentage matching based 
on the JD and the missing keywords with high accuracy.

resume: {text}
description: {jd}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}}
"""

# Streamlit app
st.title("Smart ATS")
st.text("Improve Your Resume ATS")

jd = st.text_area("Paste the Job Description")
uploaded_files = st.file_uploader("Upload Your Resumes", type="pdf", accept_multiple_files=True, help="Please upload the pdf files")

submit = st.button("Submit")

if submit:
    if uploaded_files and jd.strip():
        results = []
        for uploaded_file in uploaded_files:
            text = input_pdf_text(uploaded_file)
            response = get_gemini_repsonse(input_prompt.format(text=text, jd=jd))
            if response:
                try:
                    result = json.loads(response)  # Try parsing the response as JSON
                    results.append(result)
                except json.JSONDecodeError:
                    st.warning(f"Failed to parse response for {uploaded_file.name}. Response: {response}")
            else:
                st.warning(f"Failed to get a response for {uploaded_file.name}.")
        
        if results:
            # Convert the results to a DataFrame
            df = pd.DataFrame(results)
            st.subheader("Resume Analysis Results")
            st.dataframe(df)
    else:
        st.warning("Please provide both the job description and at least one resume.")
