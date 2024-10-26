import os
import time
import threading
import configparser
import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import html2text
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import sqlite3

from dataclasses import dataclass, field
from typing import List, Optional

from flask import Flask, request
from flask import Response as FlaskResponse

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



# 設定資料儲存路徑
DATA_DIR = Path("./data")
VECTORSTORE_PATH = DATA_DIR / "chroma_langchain_db"
SCRAPED_DATA_PATH = DATA_DIR / "scraped_data.json"

# 設置常量
ROOT_URL = 'https://www.coolenglish.edu.tw/'
LOGIN_URL = 'https://www.coolenglish.edu.tw/login/index.php'
USERNAME = ''
PASSWORD = ''
MAX_DEPTH = 3
OUTPUT_DIR = DATA_DIR / 'page_data'
# VECTORSTORE_PATH = DATA_DIR / 'normal_page'


# config檔
current_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(current_path, './config.ini'))

OPENAI_API_KEY = config['openai']['api_key']


# 應用程式設定
app = Flask(__name__)


# 全域變數
class ScrapingTask:
    def __init__(self):
        self.status = "Not Started"
        self.start_time = None
        self.end_time = None


scraping_task = ScrapingTask()

visited_urls_lock = threading.Lock()
html_urls_lock = threading.Lock()
manager = None

# 初始化嵌入和模型
embedding = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


@dataclass
class WebPage:
    root_url: str
    url: str
    title: str
    content: str
    depth: int
    parent_url: Optional[str] = None
    child_urls: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    category: str = ""

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

# 設置 chrome driver
def setup_driver():
    try:
        service = Service(ChromeDriverManager().install())
        
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-notifications")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(service=service, options=options)
        
        print("Chrome version:", driver.capabilities['browserVersion'])
        print("ChromeDriver version:", driver.capabilities['chrome']['chromedriverVersion'])
        
        return driver
    except Exception as e:
        print(f"Error in setup_driver: {str(e)}")
        raise

def verify_login(driver):
    """驗證是否成功登入"""
    try:
        # 根據登入後特有的元素來驗證，例如用戶名稱或個人資料連結
        # 這裡需要根據實際網站修改選擇器
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "usertext"))
        )
        return True
    except:
        return False

# 登入coolies存取
def save_cookies(driver, path):
    """保存cookies到文件"""
    with open(path, 'wb') as file:
        json.dump(driver.get_cookies(), file)

def load_cookies(driver, path):
    """從文件加載cookies"""
    try:
        with open(path, 'rb') as file:
            cookies = json.load(file)
            for cookie in cookies:
                driver.add_cookie(cookie)
        return True
    except:
        return False

# 解決登入問題
def login(driver):
    cookies_path = "./data/cookies.json"
    
    try:
        # 首先嘗試使用已保存的cookies
        if os.path.exists(cookies_path):
            driver.get(ROOT_URL)  # 先訪問主域名
            if load_cookies(driver, cookies_path):
                driver.get(ROOT_URL)  # 加載cookies後重新訪問
                if verify_login(driver):
                    print("使用已保存的cookies成功登入")
                    return True

        # 如果cookies無效，執行正常登入流程
        print("開始新的登入流程...")
        driver.get(LOGIN_URL)
        
        # 等待登入表單加載
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )

        # 清除可能的舊數據
        username_field = driver.find_element(By.ID, "username")
        password_field = driver.find_element(By.ID, "password")
        username_field.clear()
        password_field.clear()

        # 輸入登入信息
        username_field.send_keys(USERNAME)
        password_field.send_keys(PASSWORD)

        # 點擊登入按鈕
        login_button = driver.find_element(By.ID, "loginbtn")
        login_button.click()

        # 等待登入完成
        WebDriverWait(driver, 10).until(EC.url_contains("index"))

        # 驗證登入狀態
        if verify_login(driver):
            print("登入成功")
            # 保存新的cookies
            save_cookies(driver, cookies_path)
            return True
        else:
            print("登入可能失敗，請檢查")
            return False

    except Exception as e:
        print(f"登入過程中出現錯誤: {str(e)}")
        return False


def clean_content(content):
    # 移除多餘的空白行
    content = re.sub(r'\n\s*\n', '\n\n', content)
    return content.strip()


# url 分類
def extract_category(url):
    path = urlparse(url).path.strip('/').split('/')
    if len(path) > 0:
        return path[0].replace('-', ' ').title()
    return "Uncategorized"


# 提取summary
def extract_summary(soup, content):
    # 方法1：從meta描述提取
    description_meta = soup.find('meta', attrs={'name': 'description'})
    if description_meta:
        return description_meta['content']
    
    # 方法2：使用頁面的前幾個句子
    sentences = content.split('.')
    summary = '. '.join(sentences[:3]) + '.'
    
    # 方法3：使用NLP生成摘要（這裡使用了一個簡化的方法）
    if len(summary) > 200:
        return summary[:197] + '...'
    
    return summary


def process_url(driver, url, root_url, depth, pages, parent_url=None):
    try:

        print(f"正在爬取: {url} (深度: {depth})")
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        title = soup.title.string if soup.title else "No Title"

        if title == "404":
            return

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        content = h.handle(str(soup))
        content = clean_content(content)


        # 提取麵包屑作為keywords
        keywords = [a.text for a in soup.select('.breadcrumb a')]
        
        # 使用parent_url構建path
        path = []
        if parent_url:
            parent_page = next((p for p in pages if p.url == parent_url), None)
            if parent_page:
                path = parent_page.path + [parent_url]

        
        category = extract_category(url)
        summary = extract_summary(soup, content)

        all_links = [urljoin(root_url, a.get('href')) for a in soup.find_all('a', href=True)]
        child_urls = [link for link in all_links if link.startswith(root_url)]

        page = WebPage(
            root_url=root_url,
            url=url,
            title=title,
            content=content,
            depth=depth,
            parent_url=parent_url,
            child_urls=child_urls,
            path=path,
            keywords=keywords,
            summary=summary,
            category=category
        )

        print(f"已完成爬取: {url} (深度: {depth})")
        print(f"當前進度: 已爬取 {len(pages)} 個頁面")
        return page

    except Exception as e:
        print(f"Error in process_url for {url}: {str(e)}")


def crawl_website():
    driver = None
    visited_urls = set()
    pages = []
    url_queue = deque([(ROOT_URL, 0, None)])

    try:
        driver = setup_driver()
        
        # 執行登入並驗證
        login_success = login(driver)
        if not login_success:
            print("登入失敗，停止爬取")
            return []

        while url_queue:
            current_url, depth, parent_url = url_queue.popleft()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            driver.get(current_url)
            
            
            page = process_url(driver, current_url, ROOT_URL, depth, pages, parent_url)
            
            if page:
                pages.append(page)
                
                for child_url in page.child_urls:
                    if child_url not in visited_urls and page.depth < MAX_DEPTH:
                        url_queue.append((child_url, depth + 1, current_url))
        
        print(f"爬取完成，共爬取了 {len(pages)} 個頁面")
        return pages     

    except Exception as e:
        print(f"Error in crawl_website: {str(e)}")
        return []
    finally:
        if driver:
            driver.quit()


def initialize_database():
    conn = sqlite3.connect(os.path.join(OUTPUT_DIR, 'website_data.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pages
                 (url TEXT PRIMARY KEY, title TEXT, depth INTEGER, 
                  path TEXT, keywords TEXT, summary TEXT, category TEXT,
                  content TEXT, parent_url TEXT, child_urls TEXT)''')
    conn.commit()
    return conn

def save_pages(pages, conn):
    c = conn.cursor()
    for page in pages:
        c.execute('''INSERT OR REPLACE INTO pages 
                     (url, title, depth, path, keywords, summary, category, content, parent_url, child_urls) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (page.url, page.title, page.depth, 
                   ','.join(page.path), ','.join(page.keywords), 
                   page.summary, page.category, page.content, page.parent_url, json.dumps(page.child_urls)))
    conn.commit()

def create_sitemap(pages):
    sitemap = []
    for page in pages:
        sitemap.append({
            "url": page.url,
            "title": page.title,
            "depth": page.depth,
            "path": page.path,
            "keywords": page.keywords,
            "summary": page.summary,
            "category": page.category,
            "parent_url": page.parent_url,
            "child_urls": page.child_urls
        })
    return sitemap

def scraping_task_runner():
    global scraping_task
    scraping_task.status = "Running"
    scraping_task.start_time = datetime.now().isoformat()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pages = crawl_website()

        # 初始化並保存到數據庫
        conn = initialize_database()
        save_pages(pages, conn)

        # 創建網站地圖
        sitemap = create_sitemap(pages)
        with open(os.path.join(OUTPUT_DIR, "sitemap.json"), 'w', encoding='utf-8') as f:
            json.dump(sitemap, f, ensure_ascii=False, indent=2)

        # 保存所有內容到一個文件
        with open(os.path.join(OUTPUT_DIR, 'all_content.txt'), 'w', encoding='utf-8') as f:
            for page in pages:
                f.write(f"URL: {page.url}\n")
                f.write(f"Content:\n{page.content}\n")
                f.write("-" * 80 + "\n")

        conn.close()
        scraping_task.status = "Completed"
    except Exception as e:
        print(f"Error in scraping_task_runner: {str(e)}")
        scraping_task.status = f"Failed: {str(e)}"
    finally:
        scraping_task.end_time = datetime.now().isoformat()

# 網頁爬取功能
# def process_url(url, root_url, visited_urls, html_urls, next_queue):
#     if url in visited_urls:
#         # print("This url was visited. url:", url)
#         return

#     with visited_urls_lock:
#         if url in visited_urls:
#             # print("This url was visited (Lock). url:", url)
#             return
#         visited_urls.add(url)

#     try:
#         resp = requests.get(url, timeout=10)
#         if resp.status_code != 200:
#             print("resp.status_code != 200.")
#             return
#         if 'text/html' not in resp.headers.get('Content-Type', '').lower():
#             print(f"resp.headers not html. url:{url}")
#             return

#         with html_urls_lock:
#             html_urls.append(url)
#             print(f"Add into html_urls. url:{url}")
#             print(f"All urls :{len(html_urls)}")

#         soup = BeautifulSoup(resp.text, 'html.parser')
#         all_links = [urljoin(root_url, a.get('href')) for a in soup.find_all('a')]
#         all_links = filter(lambda x: x and x.startswith(root_url), all_links)

#         for link in all_links:
#             next_queue.put(link)

#     except requests.RequestException:
#         print(f"Failed to process {url}: {e}")
#         pass


# def bfs_website(root_url, max_workers=20):
#     visited_urls = set()
#     html_urls = []
#     queue = Queue()
#     queue.put(root_url)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         while not queue.empty():
#             next_queue = Queue()
#             futures = []
#             for _ in range(queue.qsize()):
#                 url = queue.get()
#                 future = executor.submit(
#                     process_url, url, root_url, visited_urls, html_urls, next_queue
#                 )
#                 futures.append(future)
#             for future in as_completed(futures):
#                 pass
#             queue = next_queue
#     print("Done the bfs website.")
#     return html_urls


def scrape_website():
    scraping_task.status = 'pending'
    scraping_task.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        if not SCRAPED_DATA_PATH.exists():
            root_url = 'https://www.coolenglish.edu.tw/'
            # urls = bfs_website(root_url)
            
            # 儲存爬取的 URL 列表
            with open(SCRAPED_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(urls, f, ensure_ascii=False, indent=2)
        else:
            print("Using existing scraped data.")
            with open(SCRAPED_DATA_PATH, 'r', encoding='utf-8') as f:
                urls = json.load(f)

        print(f"Processing {len(urls)} URLs")
        loader = AsyncHtmlLoader(urls)
        print("load urls.")
        docs = loader.load()
        print(f'{docs[0].page_content[:100]}')
        md_transformer = MarkdownifyTransformer()
        converted_docs = md_transformer.transform_documents(docs)
        print(f'{converted_docs[0].page_content[:100]}')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        splits = text_splitter.split_documents(converted_docs)
        print('split docs done')
        
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
        scraping_task.status = 'error'
        scraping_task.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return

    print("scraping finished!")
    scraping_task.status = 'finished'
    scraping_task.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def initialize_vectorstore():
    global manager
    if SCRAPED_DATA_PATH.exists() and VECTORSTORE_PATH.exists():
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=str(VECTORSTORE_PATH), embedding_function=embedding)
        retriever = vectorstore.as_retriever()
        manager = UserMemoryManager(retriever, llm)
        scraping_task.status = 'finished'
        scraping_task.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        scraping_task.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Vectorstore loaded successfully.")
    else:
        print("No existing vectorstore found.")
        scraping_task.status = 'not started'


def chat_with_rag(user_id, question):
    global manager
    if not manager:
        return "系統尚未準備好，請稍後再試。", []
    qa_chain = manager.get_chain_for_user(user_id)
    # result = chain({"question": question})

    # answer = result['answer']
    # source_list = [source.metadata['source'] for source in result['source_documents']]

    # return answer, source_list
    response = qa_chain.invoke(question)
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
@app.route('/api/login', methods=['GET'])
def web_login():
    
    driver = setup_driver()
    driver.get(LOGIN_URL)
    cookie = driver.get_cookies()
    with open('./data/cookies.json', 'w') as f:
        f.write(json.dumps(cookie))

    with open('./data/cookies.json', 'r') as f:
        data = json.loads(f.read())
    
    return data

@app.route('/api/start-scraping', methods=['GET'])
def start_scraping():
    
    global scraping_task
    # initialize_vectorstore()
    if scraping_task.status == "Running":
        response = {
            "description": "Scraping already in progress",
            "response": {
                "status": scraping_task.status,
                "start_time": scraping_task.start_time,
                "end_time": scraping_task.end_time
            }
        }
        return response, 400
    elif scraping_task.status == "finished":
        response = {
            "description": "Scraping already finished",
            "response": {
                "status": scraping_task.status,
                "start_time": scraping_task.start_time,
                "end_time": scraping_task.end_time
            }
        }
        return response, 400

    thread = threading.Thread(target=scraping_task_runner)
    thread.start()

    response = {
        "description": "Scraping started",
        "response": {
            "status": scraping_task.status,
            "start_time": scraping_task.start_time,
            "end_time": scraping_task.end_time
        }
    }

    return response, 200



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

    response = {
        "description": "Current scraping status",
        "response": {
            "status": scraping_task.status,
            "start_time": scraping_task.start_time,
            "end_time": scraping_task.end_time
        }
    }
    return response, 200

@app.route('/api/query', methods=['POST'])
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
    if scraping_task.status != 'finished':
        return {"message": "系統尚未準備好，請稍後再試。", "status": scraping_task.status}, 400

    data = request.json
    query_string = data.get('query_string')
    user_id = data.get('user_id')

    if not query_string or not user_id:
        return {"message": "缺少必要的參數"}, 400

    answer, sources = chat_with_rag(user_id, query_string)
    return {"answer": answer, "sources": sources}, 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)