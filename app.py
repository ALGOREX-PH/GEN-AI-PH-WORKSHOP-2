import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from utils import extract_text_from_pdf, create_chunks, SimpleChromaDB
import tempfile
warnings.filterwarnings("ignore")

# Initialize ChromaDB manager
if 'chroma_db' not in st.session_state:
    st.session_state.chroma_db = SimpleChromaDB()

generation_config = {
    "temperature": 0.1,  #  lower temperature for more focused responses
    "top_p": 0.85,
    "top_k": 40,
    "max_output_tokens": 32768,
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

SYSTEM_PROMPT = "You are a Helpful Assistant."

st.set_page_config(page_title="Introduction to Streamlit and GEMINI API", page_icon="", layout="wide")

with st.sidebar:
    api_key = st.text_input('Enter Gemini API token:', type='password')

    if api_key:
       try:
          genai.configure(api_key=api_key)
          model = genai.GenerativeModel('gemini-2.0-flash')
          response = model.generate_content("Hello") # Hello test
          st.success('Proceed to entering your prompt message!', icon='👉')
          
       except Exception as e:
          st.error(f"Invalid API key or error: {e}", icon="🚨")

    else:
       st.warning('Please enter your Gemini API token!', icon='⚠️')
    with st.container():
        l, m, r = st.columns((1, 3, 1))
        with l: st.empty()
        with m: st.empty()
        with r: st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Chat", "RAG PDF"],
        icons=['book', 'globe', 'tools', 'file-pdf'],
        menu_icon="book", 
        default_index=0,
        styles={
            "icon": {"color": "#dec960", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"}          
        })


if 'message' not in st.session_state:
    st.session_state.message = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

# Options : Home
if options == "Home" :

   st.title('This is the Home Page')
  

elif options == "About Us" :
     st.title('This is the About Us Page')
     st.write("\n")

elif options == "RAG PDF":
    st.title("PDF Knowledge Assistant")
    
    # PDF upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Process the PDF
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract text and create chunks
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(tmp_file_path)
            chunks = create_chunks(text)
            st.session_state.chroma_db.add_documents(chunks)
            st.success("PDF processed!")
        
        # Query interface
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Finding detailed answer..."):
                # Get relevant chunks with more context
                relevant_chunks = st.session_state.chroma_db.search(query, n_results=3)
                
                # Prepare detailed prompt
                context = "\n\nContext Section " + "\nContext Section ".join([f"{i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks)])
                
                prompt = f"""Based on the provided context sections from the PDF document, please provide a detailed and comprehensive answer to the question. 
                If you find relevant information in multiple context sections, synthesize them together.
                If the answer cannot be fully derived from the context, please indicate what information is missing.
                
                Context:
                {context}
                
                Question: {query}
                
                Please provide a detailed answer, explaining your reasoning and citing specific parts of the context when relevant:"""
                
                # Get response from Gemini
                response = model.generate_content(prompt)
                
                # Display response in a nice format
                st.markdown("### Answer:")
                st.markdown(response.text)
                
                # Show sources
                with st.expander("View source sections from the document"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Section {i+1}:**")
                        st.markdown(chunk)
                        st.markdown("---")

if options == 'Chat':
            if "chat_session" not in st.session_state:
                st.session_state.chat_session = model.start_chat(history=[])
                st.session_state.messages = []
                st.write("Chat session initialized.")

            if st.session_state.get("chat_session") is None:
                st.session_state.chat_session = model.start_chat(history=[])
                st.session_state.messages = []
                response = st.session_state.chat_session.send_message("You will act and introduce yourself as the following :  " + SYSTEM_PROMPT)
                with st.chat_message("assistant"):
                    st.markdown(response.text)

            for message in st.session_state.messages:
                if message['role'] == 'system':
                    continue
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_message := st.chat_input("Say something"):
                with st.chat_message("user"):
                    st.markdown(user_message)
                st.session_state.messages.append({"role": "user", "content": user_message})

                if st.session_state.get("chat_session"):
                        response = st.session_state.chat_session.send_message(user_message)
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})