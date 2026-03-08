# education-ai


# 一、项目结构

```text
ai-rag-knowledge-base/
│
├── main.py                  # FastAPI 启动入口
├── config.py                # API key 配置
├── requirements.txt
├── vector_store/
│   └── faiss_db/            # FAISS向量库存放
├── uploads/                 # 上传的PDF
├── services/
│   ├── pdf_loader.py        # PDF解析 + 文本切块
│   ├── embedding.py         # 通义千问 Embedding
│   ├── rag_chain.py         # RAG问答链
│   └── chat_memory.py       # 多轮记忆
├── web_app/
│   └── app.py               # Streamlit Web UI
├── Dockerfile               # 可选 Docker 部署
└── README.md
```

> ⚡ 这就是企业级标准结构，面试官看到就会认可你项目可落地。

---

# 二、requirements.txt

```text
fastapi
uvicorn
langchain
langchain-community
dashscope
faiss-cpu
pypdf
tiktoken
streamlit
python-multipart
```

> `python-multipart` 是上传 PDF 必须的。

---

# 三、config.py

```python
# config.py
DASHSCOPE_API_KEY = "你的通义千问APIKEY"
```

---

# 四、升级 PDF Loader（多PDF + 自动切块）

`services/pdf_loader.py`

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_pdfs(folder_path="uploads/"):
    """加载上传文件夹里的所有PDF，并切块"""
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(path)
            docs = loader.load()

            # 文本切块
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunk_docs = text_splitter.split_documents(docs)
            all_docs.extend(chunk_docs)
    return all_docs
```

✅ 支持多 PDF 自动切块

---

# 五、Embedding + FAISS（自动更新）

`services/embedding.py`

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.vectorstores import FAISS
import os

def create_or_update_vector_store(docs, db_path="vector_store/faiss_db"):
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    if os.path.exists(db_path):
        # 加载已有库并追加
        db = FAISS.load_local(db_path, embeddings)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(db_path)
    return db
```

---

# 六、RAG + 多轮记忆

`services/rag_chain.py`

```python
from langchain_community.chat_models import ChatTongyi
from langchain.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from services.chat_memory import ChatMemory

def create_qa_chain():
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    db = FAISS.load_local("vector_store/faiss_db", embeddings)
    retriever = db.as_retriever(search_kwargs={"k":3})

    llm = ChatTongyi(model_name="qwen-turbo")
    memory = ChatMemory()  # 简单多轮记忆

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=memory
    )
    return qa_chain
```

---

`services/chat_memory.py`

```python
class ChatMemory:
    """简单内存实现，用于多轮对话"""
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self):
        return self.history
```

✅ 支持多轮上下文

---

# 七、FastAPI 接口

`main.py`

```python
from fastapi import FastAPI, UploadFile
import shutil, os
from services.pdf_loader import load_pdfs
from services.embedding import create_or_update_vector_store
from services.rag_chain import create_qa_chain

app = FastAPI()
qa_chain = create_qa_chain()

@app.get("/")
def home():
    return {"message": "AI企业知识库系统运行成功"}

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_pdfs("uploads")
    create_or_update_vector_store(docs)
    return {"message": "PDF上传成功并更新知识库"}

@app.get("/ask")
def ask(q: str):
    result = qa_chain({"question": q})
    answer = result["answer"]
    return {"answer": answer}
```

---

# 八、Streamlit Web UI

`web_app/app.py`

```python
import streamlit as st
import requests

st.title("AI企业知识库问答系统")

uploaded_file = st.file_uploader("上传PDF", type="pdf")
if uploaded_file:
    files = {"file": uploaded_file}
    r = requests.post("http://127.0.0.1:8000/upload", files=files)
    st.success(r.json()["message"])

question = st.text_input("请输入你的问题")
if question:
    r = requests.get("http://127.0.0.1:8000/ask", params={"q": question})
    st.write("AI回答：", r.json()["answer"])
```

启动：

```bash
uvicorn main:app --reload
streamlit run web_app/app.py
```

---

# 九、Docker 部署

`Dockerfile`

```dockerfile
FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
