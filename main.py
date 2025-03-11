import os
import time
import threading
import configparser
import sqlite3
import json
import re
import random
from pathlib import Path
from datetime import datetime

from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
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
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify
from flask import Response as FlaskResponse

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# config檔
current_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(current_path, './config.ini'))

OPENAI_API_KEY = config['openai']['api_key']


# 設置常量
ROOT_URL = 'https://www.coolenglish.edu.tw/'
LOGIN_URL = 'https://www.coolenglish.edu.tw/login/index.php'
USERNAME = config['user']['username']
PASSWORD = config['user']['password']
MAX_DEPTH = 20
# VECTORSTORE_PATH = DATA_DIR / 'normal_page'
 
# 設定資料儲存路徑
DATA_DIR = Path("./data")
OUTPUT_DIR = DATA_DIR / 'page_data_02'
VECTORSTORE_PATH = str(DATA_DIR / "chroma_langchain_db_2")
SCRAPED_DATA_PATH = DATA_DIR / "scraped_data.json"
SQLITE_PATH = OUTPUT_DIR / "website_data.db"



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
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)
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

    def to_document(self) -> Document:
        """直接返回 Document 對象而不是字典"""
        return Document(
            page_content=self.content or '',
            metadata={
                "url": self.url or '',
                "title": self.title or '',
                "summary": self.summary or '',
                "category": self.category or '',
                "keywords": ','.join(self.keywords) if self.keywords else '',
                "path": ','.join(self.path) if self.path else '',
                "depth": self.depth if isinstance(self.depth, int) else 0,
                "parent_url": self.parent_url or ''
            }
        )
    

@dataclass
class RetrievedContext:
    """用於存儲檢索到的文檔信息"""
    url: str
    title: str
    path: List[str]
    summary: str
    content: str


class RAGChatbot:
    def __init__(self):
        """
        初始化 RAG Chatbot
        
        Args:
            sqlite_path: SQLite 數據庫路徑
            vector_store_path: Chroma 向量數據庫路徑
        """
        self.sqlite_path = SQLITE_PATH
        self.vector_store_path = VECTORSTORE_PATH
        
        # 驗證數據庫
        self._validate_databases()
        
        # 初始化 embedding 和向量數據庫
        self.embedding_model = embedding_model
        
        # 初始化 LLM
        self.llm = llm

        # 載入向量資料庫
        try:
            self.vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_model
            )
        except Exception as e:
            raise Exception(f"載入向量資料庫失敗: {str(e)}")
        
        # 初始化檢索器
        self.k = 3
        self.retriever = self.vectorstore.as_retriever(k=self.k)
        
        # 設置提示模板
        self.prompt_template = ChatPromptTemplate.from_template("""
            你是酷英網站(Cool English)的AI導覽助手 Cool，專門協助國小到高中的學生找到適合的英文學習資源。請根據以下提供的上下文來回答問題。

            上下文資訊：
            {context}

            使用者問題：
            {question}

            回答指南：
            1. 身分與語氣：
            - 使用活潑友善的口吻稱呼自己為 "Cool"
            - 對國小生：使用簡單活潑的語氣，適時加入表情符號
            - 對國中生：使用平易近人但不失專業的語氣
            - 對高中生：使用較為正式的語氣
            
            2. 回答架構：
            - 開場問候
            - 核心回答：直接針對問題重點進行回答
            - 資源整理：若有多個相關資源，條列式呈現（最多4個）
                * 資源名稱
                * 簡短說明（包含難度、特色等）
                * 完整導覽路徑
                * 直接連結
            - 使用建議：如何使用這些資源
            - 結語：鼓勵性的話語

            3. 導覽細節：
            - 提供明確的網站路徑（例如：首頁 > 課程專區 > 國小區）
            - 附上直接連結
            - 說明頁面的重要功能或注意事項
            - 若有特殊使用說明（如需全螢幕模式等），務必提醒

            4. 回答原則：
            - 確保內容的適齡性
            - 優先推薦檢索到的相關內容
            - 若有多個選擇，建議由易到難的學習順序
            - 資訊不足時，誠實告知並提供其他可能的解決方案
            
            5. 回答格式：
            - 使用明確的標題和分點
            - 重要資訊用粗體標示
            - 適當使用表情符號增加親和力
            - 保持段落清晰，避免過長

            請記住：回答要以幫助使用者找到最適合的學習資源為主要目標，同時提供清晰的導覽指引。

        """)
        
        # 建立問答鏈
        self.qa_chain = self._create_qa_chain()


    def _validate_databases(self):
        """驗證數據庫是否存在且可用"""
        if not os.path.exists(self.sqlite_path):
            raise ValueError("SQLite 數據庫不存在")
        if not os.path.exists(self.vector_store_path):
            raise ValueError("Chroma 向量數據庫不存在")

    def _format_context(self, retrieved_contexts: List[RetrievedContext]) -> str:
        """格式化上下文信息"""
        if not retrieved_contexts:
            return "找不到相關資訊。"
        
        formatted_contexts = []
        for idx, ctx in enumerate(retrieved_contexts, 1):
            
            context = f"""
            [相關頁面 {idx}]\n
            標題: {ctx.title}\n
            網址: {ctx.url}\n
            導覽路徑: {' > '.join(ctx.path)}\n
            摘要: {ctx.summary}\n
            內容: {ctx.content}\n
            """
            formatted_contexts.append(context)
            
        return "\n\n".join(formatted_contexts)

    def _process_retrieved_docs(self, docs: List) -> List[RetrievedContext]:
        """處理檢索到的文檔"""
        contexts = []
        for doc in docs:

            # 從 metadata 中獲取資訊
            context = RetrievedContext(
                url=doc.metadata.get("url", "無來源"),
                title=doc.metadata.get("title", "無標題"),
                path=doc.metadata.get("path", "").split(","),
                summary=doc.metadata.get("summary", ""),
                content=doc.page_content,
            )
            contexts.append(context)
        
        return contexts

    def _create_qa_chain(self):
        """創建問答鏈"""

        return (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def chat(self, query: str):
        """
        處理用戶查詢
        
        Args:
            query: 用戶的問題
            
        Returns:
            回答和相關來源的元組
        """
        if not query or not query.strip():
            raise ValueError("查詢不能為空")
        
        try:
            
            # 處理檢索到的文檔
            retrieved_docs = self.retriever.invoke(query)
            contexts = self._process_retrieved_docs(retrieved_docs)
            format_contexts = self._format_context(contexts)
            print(f"檢索到 {len(contexts)} 個相關頁面")
            print(contexts[0])

            # 啟動問答鏈
            response = self.qa_chain.invoke({"question": query, "context": format_contexts})
            print(f"回答: {response}")

            return response, format_contexts
            
        except Exception as e:
            raise Exception(f"聊天過程中發生錯誤: {str(e)}")
        

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
            EC.presence_of_element_located((By.CLASS_NAME, "user-menu-name"))
        )
        return True
    except:
        return False

# 登入coolies存取
def save_cookies(driver, path):
    """保存cookies到文件"""
    with open(path, 'w') as file:
        json.dump(driver.get_cookies(), file)

def load_cookies(driver, path):
    """從文件加載cookies"""
    try:
        with open(path, 'r') as file:
            cookies = json.load(file)
            for cookie in cookies:
                driver.add_cookie(cookie)

        driver.refresh()
        return True
    except:
        return False
    

def clear_cookies_and_login(driver):
    """清理 cookie 並重新登入"""
    try:
        driver.delete_all_cookies()
        time.sleep(2)  # 等待 cookie 清理完成
        return login(driver)
    except Exception as e:
        print(f"清理 cookie 時發生錯誤: {str(e)}")
        return False

def reset_driver(driver):
    """重置瀏覽器狀態"""
    try:
        driver.quit()
        driver = setup_driver()
        if not login(driver):
            print("重置後登入失敗")
            return None
        return driver
    except Exception as e:
        print(f"重置瀏覽器時發生錯誤: {str(e)}")
        return None

# 解決登入問題
def login(driver):
    cookies_path = "./data/cookies.json"
    
    try:
        # # 首先嘗試使用已保存的cookies
        # if os.path.exists(cookies_path):
        #     driver.get(ROOT_URL)  # 先訪問主域名
        #     if load_cookies(driver, cookies_path):
        #         driver.get(ROOT_URL)  # 加載cookies後重新訪問
        #         if verify_login(driver):
        #             print("使用已保存的cookies成功登入")
        #             return True

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
            load_cookies(driver, cookies_path)
            return True
        else:
            print("登入可能失敗，請檢查")
            return False

    except Exception as e:
        print(f"登入過程中出現錯誤: {str(e)}")
        return False


def should_skip_url(url):
    """檢查URL是否安全（不會導致登出）"""
    unsafe_patterns = [
        '/logout',
        'logout.php',
        'signout',
        'sign-out',
        'deauth',
        'disconnect',
        'end-session',
        '/mod/',
        '/pluginfile.php'
    ]
    
    # 轉換為小寫進行比較
    lower_url = url.lower()
    
    # 檢查是否包含不安全的模式
    return any(pattern in lower_url for pattern in unsafe_patterns)

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
    # """提取課程摘要和內容"""
    summary_parts = []
    
    # 提取 summarytext
    summary_div = soup.find('div', class_='summarytext')
    if summary_div:
        summary_parts.append(summary_div.get_text(strip=True))
    
    # 提取課程內容
    course_content = soup.find('div', class_='course-content')
    if course_content:
        accordion = course_content.find('ul', id='accordion3')
        if accordion:
            # 提取所有課程單元內容
            for item in accordion.find_all(['li', 'div']):
                text = item.get_text(strip=True)
                if text:
                    summary_parts.append(text)

    course_summary = '\n'.join(filter(None, summary_parts))

    if course_summary:
        summary = course_summary
    else: 
        # 方法2：使用頁面的前幾個句子
        sentences = content.split('.')
        summary = '. '.join(sentences[:3]) + '.'
    
    return summary


def process_url_with_retry(driver, url, root_url, depth, pages, parent_url=None, max_retries=3):
    """帶有重試機制的 process_url"""
    retries = 0
    while retries < max_retries:
        try:
            return process_url(driver, url, root_url, depth, pages, parent_url)
        except WebDriverException as e:
            if "ERR_TOO_MANY_REDIRECTS" in str(e) or "redirected you too many times" in str(e):
                print(f"遇到重定向錯誤，正在重試 ({retries + 1}/{max_retries})")
                if clear_cookies_and_login(driver):
                    retries += 1
                    continue
                
                # 如果清理 cookie 和重新登入失敗，嘗試重置瀏覽器
                driver = reset_driver(driver)
                if driver is None:
                    print("重置瀏覽器失敗，停止重試")
                    return None
                retries += 1
            else:
                print(f"遇到其他錯誤: {str(e)}")
                return None
        except Exception as e:
            print(f"處理 URL 時發生錯誤: {str(e)}")
            return None
    
    print(f"達到最大重試次數 ({max_retries})，跳過此 URL")
    return None

def process_url(driver, url, root_url, depth, pages, parent_url=None):
    try:

        # 首先檢查URL是否安全
        if should_skip_url(url):
            print(f"跳過不安全的URL: {url}")
            return None

        print(f"正在爬取: {url} (深度: {depth})")
        driver.get(url)
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception as e:
            print(f"等待頁面加載超時: {str(e)}")
            return None

        # 檢查是否有錯誤頁面標識
        if "ERR_TOO_MANY_REDIRECTS" in driver.page_source:
            raise WebDriverException("ERR_TOO_MANY_REDIRECTS")

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
        child_urls = [link for link in all_links if link.startswith(root_url) and not should_skip_url(link)]

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
    error_count = 0
    max_errors = 5  # 最大連續錯誤次數

    try:
        driver = setup_driver()
        print("開始登錄...")
        if not login(driver):
            print("初始登入失敗")
            return []
        print("登錄成功")

        while url_queue:
            current_url, depth, parent_url = url_queue.popleft()
            
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)            
            
            page = process_url_with_retry(driver, current_url, ROOT_URL, depth, pages, parent_url)
            
            if page:
                error_count = 0  # 重置錯誤計數
                pages.append(page)
                
                if page.depth < MAX_DEPTH:
                    for child_url in page.child_urls:
                        if child_url not in visited_urls:
                            url_queue.append((child_url, depth + 1, current_url))
            else:
                error_count += 1
                print(f"連續錯誤次數: {error_count}")

            # 每爬取 100 個頁面後強制清理一次 cookie
            if len(pages) % 200 == 0:
                print("定期清理 cookie 並重新登入")
                clear_cookies_and_login(driver)
            
            # 添加延遲，避免請求過於頻繁
            time.sleep(random.uniform(1, 3))
        
        print(f"爬取完成，共爬取了 {len(pages)} 個頁面")
        return pages     

    except Exception as e:
        print(f"Error in crawl_website: {str(e)}")
        return []
    finally:
        if driver:
            driver.quit()


def initialize_database():
    conn = sqlite3.connect(SQLITE_PATH)
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


def get_webpages_from_db() -> List[WebPage]:
    """從數據庫獲取WebPage對象列表"""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    
    # 修改SQL查詢以匹配WebPage的所有字段
    c.execute('''
        SELECT url, title, content, depth, parent_url, 
               child_urls, path, keywords, summary, category
        FROM pages
    ''')
    
    webpages = []
    for row in c.fetchall():
        (url, title, content, depth, parent_url, 
         child_urls, path, keywords, summary, category) = row
        
        try:
            # child_urls 是JSON格式
            child_urls = json.loads(child_urls) if child_urls else []
            # path 和 keywords 是逗號分隔的字符串
            path = path.split(',') if path else []
            keywords = keywords.split(',') if keywords else []
        except (json.JSONDecodeError, AttributeError):
            child_urls = []
            path = []
            keywords = []
        
        
        webpage = WebPage(
            root_url=ROOT_URL,
            url=url,
            title=title,
            content=content,
            depth=depth,
            parent_url=parent_url,
            child_urls=child_urls,
            path=path,
            keywords=keywords,
            summary=summary or "",
            category=category or ""
        )
        webpages.append(webpage)
    
    conn.close()
    print(len(webpages))
    return webpages


def convert_to_documents() -> List[Document]:
    """將WebPage對象轉換為Document對象"""
    try:
        webpages = get_webpages_from_db()
        
        # 數據清理和驗證
        cleaned_documents = []
        for webpage in webpages:
            try:
                if len(webpage.content.strip()) > 0:  # 確保內容不為空
                    doc = webpage.to_document()
                    cleaned_documents.append(doc)
            except Exception as e:
                print(f"Error processing webpage {webpage.url}: {str(e)}")
                continue
        
        if not cleaned_documents:
            raise ValueError("No valid documents were created")
            
        print(f"Successfully created {len(cleaned_documents)} documents")
        return cleaned_documents
        
    except Exception as e:
        print(f"Error in convert_to_documents: {str(e)}")
        raise


def create_vector_store(documents: List[Document]):
    """創建向量數據庫"""
    try:
        if not documents:
            raise ValueError("No documents provided for vector store creation")
            
        # 文本分割
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=256
            )
            split_docs = text_splitter.split_documents(documents)
            print(f"Successfully split documents into {len(split_docs)} chunks")
        except Exception as e:
            raise Exception(f"Error during text splitting: {str(e)}")

        print(f"Using vector store path: {VECTORSTORE_PATH}")
        
        # 創建向量數據庫
        try:
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_model,
                persist_directory=str(VECTORSTORE_PATH)  # 確保是字符串
            )
            print("Successfully created vector store")
            return vectorstore
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
            
    except Exception as e:
        print(f"Error in create_vector_store: {str(e)}")
        raise


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


# def scrape_website():
#     scraping_task.status = 'pending'
#     scraping_task.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     try:
#         if not SCRAPED_DATA_PATH.exists():
#             root_url = 'https://www.coolenglish.edu.tw/'
#             # urls = bfs_website(root_url)
            
#             # 儲存爬取的 URL 列表
#             with open(SCRAPED_DATA_PATH, 'w', encoding='utf-8') as f:
#                 json.dump(urls, f, ensure_ascii=False, indent=2)
#         else:
#             print("Using existing scraped data.")
#             with open(SCRAPED_DATA_PATH, 'r', encoding='utf-8') as f:
#                 urls = json.load(f)

#         print(f"Processing {len(urls)} URLs")
#         loader = AsyncHtmlLoader(urls)
#         print("load urls.")
#         docs = loader.load()
#         print(f'{docs[0].page_content[:100]}')
#         md_transformer = MarkdownifyTransformer()
#         converted_docs = md_transformer.transform_documents(docs)
#         print(f'{converted_docs[0].page_content[:100]}')
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
#         splits = text_splitter.split_documents(converted_docs)
#         print('split docs done')
        
#         print(f"Processing {len(splits)} splits")
#         print("Creating embeddings and vectorstore. This may take a while...")
#         vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=str(VECTORSTORE_PATH))
#         print("Embeddings created, persisting vectorstore")

#         print("Creating retriever")
#         retriever = vectorstore.as_retriever()
#         print("Initializing UserMemoryManager")
#         global manager
#         manager = UserMemoryManager(retriever, llm)
#     except Exception as e:
#         print("Error: ", e)
#         scraping_task.status = 'error'
#         scraping_task.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         return

#     print("scraping finished!")
#     scraping_task.status = 'finished'
#     scraping_task.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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

# 全局 chatbot 實例
chatbot = None

def initialize_chatbot():
    """初始化 chatbot"""
    global chatbot
    try:
        chatbot = RAGChatbot()
        return True
    except Exception as e:
        print(f"Chatbot 初始化失敗: {str(e)}")
        return False



# 定義 API 路由
@app.route('/api/login', methods=['GET'])
def web_login():
    
    driver = setup_driver()
    login_success = login(driver)
    if not login_success:
        print("登入失敗，停止爬取")
        return []

    driver.get('https://www.coolenglish.edu.tw/course/view.php?id=115')
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    title = soup.title.string if soup.title else "No Title"

    if title == "404":
        return

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    content = h.handle(str(soup))

    
    return content

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

@app.route('/api/initialize', methods=['GET'])
def initialize():
    """初始化 API"""
    if initialize_chatbot():
        return jsonify({"message": "Chatbot 初始化成功"}), 200
    return jsonify({"error": "Chatbot 初始化失敗"}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """查詢 API"""
    if not chatbot:
        return jsonify({"error": "Chatbot 尚未初始化"}), 503

    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "缺少查詢內容"}), 400

        answer, contexts = chatbot.chat(data['query'])
        
        # 格式化回應
        response_data = {
            "answer": answer,
            "contexts": contexts
        }
        
        return response_data, 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/vectorstore/create', methods=['GET'])
def create_vectorstore_endpoint():
    try:
        # 檢查向量資料庫目錄是否存在
        if os.path.exists(VECTORSTORE_PATH):
            return jsonify({
                "status": "error",
                "message": "Vector store already exists"
            }), 400

        # 獲取文檔
        documents = convert_to_documents()
        
        if not documents:
            return jsonify({
                "status": "error",
                "message": "No documents found in database"
            }), 400

        # 創建向量資料庫
        vectorstore = create_vector_store(documents)
        
        return jsonify({
            "status": "success",
            "message": "Vector store created successfully",
            "document_count": len(documents)
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/vectorstore/search', methods=['POST'])
def search_vectorstore():
    try:
        data = request.get_json()
        query = data.get('query')
        k = data.get('k', 3)  # 默認返回3個結果

        if not query:
            return jsonify({
                "status": "error",
                "message": "Query is required"
            }), 400

        # 載入現有的向量資料庫
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embedding_model
        )

        # 搜索相似文檔
        results = vectorstore.similarity_search(query, k=k)
        
        # 格式化結果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return jsonify({
            "status": "success",
            "results": formatted_results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/vectorstore/status', methods=['GET'])
def get_vectorstore_status():
    try:
        exists = os.path.exists(VECTORSTORE_PATH)
        
        if not exists:
            return jsonify({
                "status": "not_found",
                "message": "Vector store does not exist"
            })

        # 嘗試載入向量資料庫以確認其可用性
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embedding_model
        )
        
        return jsonify({
            "status": "available",
            "message": "Vector store is ready",
            "location": VECTORSTORE_PATH
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)