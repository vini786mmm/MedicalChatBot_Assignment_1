import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize Groq LLM (e.g. LLaMA 3)
llm = ChatGroq(
    model="llama3-8b-8192",  # You can also use "mixtral-8x7b-32768" or "gemma-7b-it"
    temperature=0.7,
    max_tokens=1024
)

# Prompt Template
prompt_template = """
As a highly knowledgeable medical assistant, your role is to accurately interpret medical queries and 
provide responses using our specialized medical database. Follow these directives to ensure optimal user interactions:
1. Precision in Answers: Respond solely with information directly relevant to the user's query from our medical database.
2. Topic Relevance: Limit your expertise to:
   - Medical Prescriptions
   - Medical Advice
   - Symptom Analysis
   - Treatment Recommendations
   - Preventive Healthcare
3. If question is off-topic, politely refuse.
4. Promote accurate, up-to-date health practices.

Medical Context:
{context}

Question:
{question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Retrieval-based QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    result = rag_chain.invoke({"query": question})
    response_text = result.get("result", "").strip()

    if "Answer:" in response_text:
        return response_text.split("Answer:")[-1].strip()
    return response_text

# --- Streamlit UI starts here ---
st.markdown("""
    <style>
        .appview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align: left; color: white; padding-top: 35px; border-bottom: 3px solid red;'>
    Discover the AI Medical Recommendations 💉🩺
</h3>""", unsafe_allow_html=True)
st.write('Made By Vinti Singh')

with st.sidebar:
    st.title('🤖 MedBot: Your AI Health Companion')
    st.markdown("""
Hi! 👋 I'm here to help with your medical queries and prescriptions. Try:
- General Health Tips 🏥
- Symptom Analysis 🤒
- Medication Info 💊
- Preventive Care 🛡️
""")

initial_message = """
Hi there! I'm your MedBot 🤖  
Here are some things you can ask:
- 🩺 Suggest tablets for "<symptom>"  
- 🩺 How to boost immunity?  
- 🩺 What are the side effects of "<tablet>"?
"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the latest medical advice for you..."):
            response = get_response(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
