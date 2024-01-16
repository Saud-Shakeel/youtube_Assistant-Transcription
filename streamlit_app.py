import streamlit as st
import langchain_helper as lch
import textwrap


st.title('Youtube Video Assistant/Transcription')

with st.sidebar:
    with st.form(key='form'):
        youtube_url = st.sidebar.text_area(label='Enter the Youtube Video URL ?', max_chars=50)
        user_query = st.sidebar.text_area(label='Ask me About the Video ?', max_chars=80, key='query')

        submit_btn = st.form_submit_button(label='submit')

if user_query:         
    # db = lch.create_vector_db_from_youtube_url(youtube_url)
    db = lch.return_db_embeddings()
    response = lch.get_response_from_query(db, user_query)
    st.subheader('Answer:')
    st.text(textwrap.fill(response['query-text'], width=80))




        