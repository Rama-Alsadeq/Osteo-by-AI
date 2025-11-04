# chatbot/rag.py
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

# المسارات
BASE_DIR = Path(__file__).resolve().parent
SOURCES_DIR = BASE_DIR / "data" / "project_docs" 
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "project_docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# تنظيف قاعدة بيانات قديمة
if CHROMA_DIR.exists():
    print("🗑️ حذف قاعدة البيانات القديمة...")
    shutil.rmtree(CHROMA_DIR)

# تحميل المستندات
def load_documents():
    loaders = [
        DirectoryLoader(str(SOURCES_DIR), glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(str(SOURCES_DIR), glob="**/*.html", loader_cls=UnstructuredHTMLLoader),
    ]

    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            print(f"✅ Loader {loader.__class__.__name__} حمل {len(docs)} مستندات")
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ خطأ أثناء تحميل الملفات: {e}")

    print(f"📄 إجمالي المستندات المحملة: {len(documents)}")
    return documents

# تقسيم النصوص
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✂️ تم تقسيم المستندات إلى {len(chunks)} جزء/أجزاء")
    return chunks

# بناء قاعدة Chroma
def build_vector_db():
    print("📁 تحميل المستندات...")
    documents = load_documents()

    print("🔢 تقسيم المستندات...")
    chunks = split_documents(documents)

    print("💾 إنشاء تمثيلات (Embeddings)...")
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    print("💾 بناء قاعدة Chroma...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )

    print("✅ تم بناء قاعدة المعرفة بنجاح!")

if __name__ == "__main__":
    build_vector_db()
