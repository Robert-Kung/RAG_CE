import os
import time
import threading
import configparser
import sqlite3
import json
from pathlib import Path
from queue import Queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from flask import Flask, request
from flask import Response as FlaskResponse

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定資料儲存路徑
DATA_DIR = Path("./data")
VECTORSTORE_PATH = DATA_DIR / "chroma_langchain_db"
SCRAPED_DATA_PATH = DATA_DIR / "scraped_data.json"

# 確保資料目錄存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# config檔
current_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(current_path, './config.ini'))

OPENAI_API_KEY = config['openai']['api_key']


# 應用程式設定
app = Flask(__name__)


# 全域變數
scraping_status = {
    'status': 'not started',
    'start_time': '',
    'end_time': ''
}
visited_urls_lock = threading.Lock()
html_urls_lock = threading.Lock()
manager = None

# 初始化嵌入和模型
embedding = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


# 用戶記憶管理類
class UserMemoryManager:
    def __init__(self, retriever, llm, memory_window=5, inactive_time=300):
        self.retriever = retriever
        self.llm = llm
        self.memory_window = memory_window
        self.inactive_time = inactive_time
        self.user_memories = {}
        self.last_activity = {}
        self.lock = threading.Lock()

    def get_chain_for_user(self, user_id):
        with self.lock:
            # if user_id not in self.user_memories:
            #     memory = ConversationBufferWindowMemory(
            #         memory_key="chat_history",
            #         return_messages=True,
            #         output_key='answer'
            #     )
            #     self.user_memories[user_id] = memory
            # else:
            #     memory = self.user_memories[user_id]

            # self.last_activity[user_id] = datetime.now()

            # return ConversationalRetrievalChain.from_llm(
            #     llm=self.llm,
            #     retriever=self.retriever,
            #     memory=memory,
            #     return_source_documents=True,
            # )

            RAG_TEMPLATE = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            <context>
            {context}
            </context>

            Answer the following question:

            {question}"""

            rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

            qa_chain = (
                {"context": self.retriever , "question": RunnablePassthrough()}
                | rag_prompt
                | self.llm
                | StrOutputParser()
            )

            return qa_chain

    def clean_inactive_memories(self):
        with self.lock:
            current_time = datetime.now()
            inactive_users = [
                user_id for user_id, last_active in self.last_activity.items()
                if (current_time - last_active).total_seconds() > self.inactive_time
            ]
            for user_id in inactive_users:
                del self.user_memories[user_id]
                del self.last_activity[user_id]
            return len(inactive_users)


# 網頁爬取功能
def process_url(url, root_url, visited_urls, html_urls, next_queue):
    if url in visited_urls:
        # print("This url was visited. url:", url)
        return

    with visited_urls_lock:
        if url in visited_urls:
            # print("This url was visited (Lock). url:", url)
            return
        visited_urls.add(url)

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print("resp.status_code != 200.")
            return
        if 'text/html' not in resp.headers.get('Content-Type', '').lower():
            print(f"resp.headers not html. url:{url}")
            return

        with html_urls_lock:
            html_urls.append(url)
            print(f"Add into html_urls. url:{url}")
            print(f"All urls :{len(html_urls)}")

        soup = BeautifulSoup(resp.text, 'html.parser')
        all_links = [urljoin(root_url, a.get('href')) for a in soup.find_all('a')]
        all_links = filter(lambda x: x and x.startswith(root_url), all_links)

        for link in all_links:
            next_queue.put(link)

    except requests.RequestException:
        print(f"Failed to process {url}: {e}")
        pass


def bfs_website(root_url, max_workers=20):
    visited_urls = set()
    html_urls = []
    queue = Queue()
    queue.put(root_url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while not queue.empty():
            next_queue = Queue()
            futures = []
            for _ in range(queue.qsize()):
                url = queue.get()
                future = executor.submit(
                    process_url, url, root_url, visited_urls, html_urls, next_queue
                )
                futures.append(future)
            for future in as_completed(futures):
                pass
            queue = next_queue
    print("Done the bfs website.")
    return html_urls


def scrape_website():
    scraping_status['status'] = 'pending'
    scraping_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        if not SCRAPED_DATA_PATH.exists():
            root_url = 'https://www.coolenglish.edu.tw/'
            urls = bfs_website(root_url)
            
            # 儲存爬取的 URL 列表
            with open(SCRAPED_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(urls, f, ensure_ascii=False, indent=2)
        else:
            print("Using existing scraped data.")
            with open(SCRAPED_DATA_PATH, 'r', encoding='utf-8') as f:
                urls = json.load(f)

        print(f"Processing {len(urls)} URLs")
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        md_transformer = MarkdownifyTransformer()
        converted_docs = md_transformer.transform_documents(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        splits = text_splitter.split_documents(converted_docs)
        
        print(f"Processing {len(splits)} splits")
        print("Creating embeddings and vectorstore. This may take a while...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=str(VECTORSTORE_PATH))
        print("Embeddings created, persisting vectorstore")

        print("Creating retriever")
        retriever = vectorstore.as_retriever()
        print("Initializing UserMemoryManager")
        global manager
        manager = UserMemoryManager(retriever, llm)
    except Exception as e:
        print("Error: ", e)
        scraping_status['status'] = 'error'
        scraping_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return

    print("scraping finished!")
    scraping_status['status'] = 'finished'
    scraping_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def initialize_vectorstore():
    global manager
    if SCRAPED_DATA_PATH.exists() and VECTORSTORE_PATH.exists():
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=str(VECTORSTORE_PATH), embedding_function=embedding)
        retriever = vectorstore.as_retriever()
        manager = UserMemoryManager(retriever, llm)
        scraping_status['status'] = 'finished'
        print("Vectorstore loaded successfully.")
    else:
        print("No existing vectorstore found.")
        scraping_status['status'] = 'not started'


def chat_with_rag(user_id, question):
    global manager
    if not manager:
        return "系統尚未準備好，請稍後再試。", []
    qa_chain = manager.get_chain_for_user(user_id)
    # result = chain({"question": question})

    # answer = result['answer']
    # source_list = [source.metadata['source'] for source in result['source_documents']]

    # return answer, source_list
    response = qa_chain.invoke("介紹這個網站")
    source = "None"
    return response, source


def periodic_cleanup():
    global manager
    while True:
        time.sleep(60)
        if manager:
            cleaned = manager.clean_inactive_memories()
            if cleaned > 0:
                print(f"已清理 {cleaned} 個不活躍使用者的記憶")


cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()


# 定義 API 路由
@app.route('/api/start-scraping', methods=['GET'])
def start_scraping():
    """
    start scrapying
    ---
    tags:
      - retrieval
    responses:
      200:
        description: start scrapying
        schema:
          id: scrapying_status
          properties:
            description:
              type: string
            response:
              properties:
                status:
                  type: string
                start_time:
                  type: string
                end_time:
                  type: string
      400:
        description: scrapying is pending
    """
    
    if scraping_status['status'] == 'pending':
        return {"message": "爬取作業正在進行中", "status": scraping_status}, 400

    scraping_thread = threading.Thread(target=scrape_website)
    scraping_thread.start()

    return {"message": "已開始爬取網站", "status": scraping_status}, 200


@app.route('/api/scraping-status', methods=['GET'])
def check_scraping_status():
    """
    check the status of scrapying
    ---
    tags:
      - retrieval
    responses:
      200:
        description: scrapying status
        schema:
          id: scrapying_status
          properties:
            description:
              type: string
            response:
              properties:
                status:
                  type: string
                start_time:
                  type: string
                end_time:
                  type: string
    """
    return {"message": "查詢狀態成功", "status": scraping_status}, 200


@app.route('/api/query', methods=['GET'])
def query():
    """
    chat retrieval augmented generation
    ---
    tags:
      - retrieval
    parameters:
      - name: query_string
        in: query
        description: query string
        required: true
        type: string
      - name: person_id
        in: query
        description: person who can multi-turn conversations
        required: true
        type: string
    responses:
      200:
        description: chat retrieval augmented generation
      400:
        description: scrapying is not ready
    """
    if scraping_status['status'] != 'finished':
        return {"message": "系統尚未準備好，請稍後再試。", "status": scraping_status}, 400

    query_string = request.args.get('query_string')
    user_id = request.args.get('user_id')

    if not query_string or not user_id:
        return {"message": "缺少必要的參數"}, 400

    answer, sources = chat_with_rag(user_id, query_string)
    return {"answer": answer, "sources": sources}, 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)