import streamlit as st
import os
import time
from agents import generate_blog_post  # Import function to interact with agents
from crewai_tools import PDFSearchTool 
from langchain_community.tools import DuckDuckGoSearchRun 

# Streamlit app configuration
st.set_page_config(page_title="Blogpost Generator with Multi-Agent System")

# Streamlit app title
st.title("Multi-Agent Blogpost Generator")

# Attempt at formatting
st.markdown("""
<style>
.text-container {
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    overflow-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs for API Key and Model Selection
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password', help="Enter your OpenAI API key.")
GROQ_API_KEY = st.sidebar.text_input('Groq API Key (optional)', type='password', help="Enter your Groq API key if available.")

# Validate and set OpenAI API Key if provided
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'  # Set default model
    st.sidebar.success("OpenAI API Key Set Successfully!")
else:
    st.sidebar.warning("Please provide your OpenAI API Key to enable AI-driven content generation.")

# Ensure the Groq API key is optional and handled accordingly
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    st.sidebar.success("Groq API Key Set Successfully!")

# User input for company and topic
company = st.text_input("Company Name", value="theDevMasters")
location = st.text_input("Company Location", value="Irvine, California")
topic = st.text_input("Blog Topic", value="Artificial Intelligence in Real Estate, Fashion, and Healthcare")

# PDF Upload Option
uploaded_file = st.file_uploader("Upload a PDF for research", type="pdf")

# Optimization Choice
optimize_for = st.selectbox("Optimize Blog for", options=["SEO", "GEO"], index=0)

# Initialize tools if PDF is uploaded
if uploaded_file:
    # Save the uploaded PDF to a local file
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize the PDF search tool with the uploaded PDF
    pdf_search_tool = PDFSearchTool(pdf='uploaded_document.pdf')
    st.success("PDF uploaded and initialized successfully!")
else:
    st.warning("Please upload a PDF file for the research.")

# Initialize DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Button to start the process
if st.button("Generate Blog Post"):
    if not OPENAI_API_KEY:
        st.error("Please provide your OpenAI API Key!")
    elif not uploaded_file:
        st.error("Please upload a PDF file for research!")
    else:
        with st.spinner("Generating blog post..."):
            # Generate blog post using the multi-agent system
            result = generate_blog_post(company, location, topic, optimize_for, search_tool, pdf_search_tool)
            st.success("Blog post generated successfully!")

            # Display result
            st.subheader("Generated Blog Post")
            result_text = "\n".join([task.raw for task in result.tasks_output])
            st.markdown(result_text)
