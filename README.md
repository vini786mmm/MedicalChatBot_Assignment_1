
# 🩺 MedBot: AI-Powered Medical Chatbot

**MedBot** is a Retrieval-Augmented Generation (RAG)–based AI assistant designed to answer medical questions using a reference medical knowledge base. It uses state-of-the-art language models, embedding techniques, and vector storage to deliver accurate and context-aware responses on topics like symptoms, medications, diagnostics, treatments, and preventive healthcare.

---

## 📂 Project Structure

```
├── medicalbot.py             # Streamlit UI + RAG QA pipeline
├── vector_embedding.py       # PDF loader, text splitter, and Chroma vector embedding setup
├── requirements.txt          # Python dependencies
├── Medical_book.pdf          # Knowledge base: Gale Encyclopedia of Medicine
├── data/                     # Persisted ChromaDB vector store
└── .env                      # API keys and environment variables (not included)
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vini786mmm/MedicalChatBot_Assignment_1.git
cd Rag_Project_Assignment_2
```

### 2. Create a `.env` File

Create a `.env` file in the root directory with the following contents:

```env
GROQ_API_KEY=your_groq_api_key
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Build the Knowledge Base

To embed the content from `Medical_book.pdf` into a vector database:

```bash
python vector_embedding.py
```

This will generate vector embeddings and persist them in the `data/` directory using ChromaDB.

---

## 🚀 Run the Chatbot

```bash
streamlit run medicalbot.py
```

Then open your browser and go to url.

---

## 💬 Example Queries

- 🩺 *Suggest tablets for headache*
- 🩺 *What are the side effects of paracetamol?*
- 🩺 *How to boost immunity naturally?*
- 🩺 *What does abdominal ultrasound detect?*

---

## 🧠 Technologies Used

- **LLM**: [Groq LLaMA 3](https://www.groq.com/)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **UI Framework**: [Streamlit](https://streamlit.io/)
- **RAG Pipeline**: [LangChain](https://www.langchain.com/)
- **Medical Data Source**: *Gale Encyclopedia of Medicine*

---

## ⚠️ Disclaimer

This project is intended for educational purposes only and does **not** constitute professional medical advice. Always consult a licensed medical practitioner for diagnosis or treatment.

---

## 🙌 Author

**Vinti Singh**  
*MedBot: Your AI Health Companion*
