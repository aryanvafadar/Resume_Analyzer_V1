import streamlit as st
import classes

# Page Title, Page Config and Instantiate our class objects
st.set_page_config(
    page_icon='Resume Analyzer',
    layout='wide', 
    initial_sidebar_state='collapsed'
)
st.header('Resume Analyzer')
fu = classes.FileUploader()
st.divider()

# User resume upload section
resume_section, job_posting_section = st.columns([4, 6], gap='large')
with resume_section:
    fu.get_resume_file()
with job_posting_section:
    fu.get_job_posting_file()
st.divider()

similarity_section = st.columns(1)
visual_section, missing_keywords = st.columns([7, 3])

# Analyze section
with similarity_section[0]:
    fu.vectorize_resume_and_job_posting()
    st.divider()

# Display keyword matches
with visual_section:
    fu.display_matches_and_keywords_between_resume_and_job_posting()

# Highlight Missing Keywords
with missing_keywords:
    fu.highlight_mising_keywords_in_resume()