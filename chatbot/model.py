from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "project_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_answer(query: str) -> str:
    """
    استدعاء LLM مع استرجاع المعلومات من قاعدة RAG (Chroma)
    """
    # إعداد embeddings و Chroma
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embed_model,
        collection_name=COLLECTION_NAME
    )

    # إنشاء retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 17})

    # استرجاع المستندات
    try:
        docs = retriever.get_relevant_documents(query)
    except AttributeError:

        docs = retriever._get_relevant_documents(query, run_manager=None)

    # دمج محتوى المستندات في سياق واحد
    context = "\n".join([d.page_content for d in docs])

    # إعداد LLM مع Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # قالب السؤال
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant for the academic graduation project "Osteo by AI",
    which diagnoses osteoarthritis and osteoporosis using AI models.

    Instructions for answering:
    1. Use ONLY the information provided in the project documents (context). Do not invent answers.
    2. Answer clearly and concisely in a way understandable by both technical and non-technical readers
     and avoid overly long or repetitive explanations.
    3. Consider all types of information in the documents:
       - Image-based AI models (X-ray analysis) for osteoarthritis and osteoporosis
       - Numerical/clinical AI models (risk factors, Lifestyle Submodel , BMD)
       - Rule-based methods or clinical decision logic (T-score, Z-score)
    4. If the question is about a disease (symptoms, diagnosis, progression), provide:
       - How the model assesses or predicts it
       - Relevant thresholds or criteria mentioned
       - Any metrics or confidence levels if available
    5. If the question is about model performance, provide:
       - Accuracy, precision, recall, F1-score, or any metric mentioned
    6. If multiple models or approaches are used, summarize each separately, then provide overall conclusions if documented.
    7. Mention explicitly if the requested information is not present in the documents.
    8. Include references to the document type (docx, html) or page/chunk if it is requested.
    9. Only provide explanations related to the project's AI models, results, or methodology.
    10. Do NOT give any instructions or technical details about the old website (Flask + HTML), 
  as the new website uses Streamlit and you have not been trained on its implementation.
  11. You can mention the new website generally (e.g., "The project will be available as a Streamlit app") 
  but avoid any step-by-step instructions.


    Context from project documents:
    {context}

    Question: {question}
    """)

    # تحضير prompt وطلب الإجابة من LLM
    final_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)
    return response.content
