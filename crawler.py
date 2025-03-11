import os
import time
import random
import json
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

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


@dataclass
class WebPage:
    """網頁資料模型，用於儲存爬取的網頁內容和相關資訊"""
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
    html: str = ""  # 新增存儲原始HTML


class WebsiteCrawler:
    """網站爬蟲類，負責爬取網站內容"""
    
    def __init__(self, root_url, login_url, username, password, max_depth=20, output_dir=None):
        """
        初始化爬蟲
        
        Args:
            root_url: 網站根URL
            login_url: 登入頁面URL
            username: 登入用戶名
            password: 登入密碼
            max_depth: 最大爬取深度
            output_dir: 輸出目錄路徑
        """
        self.root_url = root_url
        self.login_url = login_url
        self.username = username
        self.password = password
        self.max_depth = max_depth
        self.output_dir = output_dir or Path("./data/page_data")
        self.cookies_path = Path("./data/cookies.json")
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 爬蟲狀態
        self.status = "Not Started"
        self.start_time = None
        self.end_time = None
        
        # 爬取結果
        self.visited_urls = set()
        self.pages = []
        
        # 初始化driver為None，在需要時才創建
        self.driver = None

    def setup_driver(self):
        """
        設置Chrome瀏覽器驅動
        
        Returns:
            webdriver.Chrome: 設置好的Chrome驅動實例
        """
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
            print(f"設置驅動時出錯: {str(e)}")
            raise

    def save_cookies(self, path):
        """
        保存cookies到文件
        
        Args:
            path: cookies保存路徑
        """
        with open(path, 'w') as file:
            json.dump(self.driver.get_cookies(), file)

    def load_cookies(self, path):
        """
        從文件加載cookies
        
        Args:
            path: cookies文件路徑
            
        Returns:
            bool: 是否成功加載cookies
        """
        try:
            with open(path, 'r') as file:
                cookies = json.load(file)
                for cookie in cookies:
                    self.driver.add_cookie(cookie)

            self.driver.refresh()
            return True
        except Exception as e:
            print(f"加載cookies失敗: {str(e)}")
            return False

    def verify_login(self):
        """
        驗證是否成功登入
        
        Returns:
            bool: 是否已登入
        """
        try:
            # 根據登入後特有的元素來驗證，例如用戶名稱或個人資料連結
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "user-menu-name"))
            )
            return True
        except Exception as e:
            print(f"驗證登入失敗: {str(e)}")
            return False

    def clear_cookies_and_login(self):
        """
        清理cookie並重新登入
        
        Returns:
            bool: 是否成功登入
        """
        try:
            self.driver.delete_all_cookies()
            time.sleep(2)  # 等待cookie清理完成
            return self.login()
        except Exception as e:
            print(f"清理cookie時發生錯誤: {str(e)}")
            return False

    def reset_driver(self):
        """
        重置瀏覽器狀態
        
        Returns:
            webdriver.Chrome: 重置後的驅動，如果失敗則返回None
        """
        try:
            if self.driver:
                self.driver.quit()
            self.driver = self.setup_driver()
            if not self.login():
                print("重置後登入失敗")
                return None
            return self.driver
        except Exception as e:
            print(f"重置瀏覽器時發生錯誤: {str(e)}")
            return None

    def login(self):
        """
        登入網站
        
        Returns:
            bool: 是否成功登入
        """
        try:
            # 先嘗試使用已保存的cookies
            if os.path.exists(self.cookies_path):
                self.driver.get(self.root_url)  # 先訪問主域名
                if self.load_cookies(self.cookies_path):
                    self.driver.get(self.root_url)  # 加載cookies後重新訪問
                    if self.verify_login():
                        print("使用已保存的cookies成功登入")
                        return True

            # 如果cookies無效，執行正常登入流程
            print("開始新的登入流程...")
            self.driver.get(self.login_url)
            
            # 等待登入表單加載
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )

            # 清除可能的舊數據
            username_field = self.driver.find_element(By.ID, "username")
            password_field = self.driver.find_element(By.ID, "password")
            username_field.clear()
            password_field.clear()

            # 輸入登入信息
            username_field.send_keys(self.username)
            password_field.send_keys(self.password)

            # 點擊登入按鈕
            login_button = self.driver.find_element(By.ID, "loginbtn")
            login_button.click()

            # 等待登入完成
            WebDriverWait(self.driver, 10).until(EC.url_contains("index"))

            # 驗證登入狀態
            if self.verify_login():
                print("登入成功")
                # 保存新的cookies
                self.save_cookies(self.cookies_path)
                return True
            else:
                print("登入可能失敗，請檢查")
                return False

        except Exception as e:
            print(f"登入過程中出現錯誤: {str(e)}")
            return False

    def should_skip_url(self, url):
        """
        檢查URL是否應該被跳過（不安全或不需要爬取）
        
        Args:
            url: 要檢查的URL
            
        Returns:
            bool: 是否應該跳過
        """
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

    def clean_content(self, content):
        """
        清理內容，移除多餘的空白行等
        
        Args:
            content: 原始內容
            
        Returns:
            str: 清理後的內容
        """
        import re
        # 移除多餘的空白行
        content = re.sub(r'\n\s*\n', '\n\n', content)
        return content.strip()

    def extract_category(self, url):
        """
        從URL中提取類別
        
        Args:
            url: 網頁URL
            
        Returns:
            str: 提取的類別
        """
        path = urlparse(url).path.strip('/').split('/')
        if len(path) > 0:
            return path[0].replace('-', ' ').title()
        return "Uncategorized"

    def extract_summary(self, soup, content):
        """
        提取網頁摘要
        
        Args:
            soup: BeautifulSoup對象
            content: 網頁內容
            
        Returns:
            str: 提取的摘要
        """
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

    def process_url(self, url, depth, parent_url=None):
        """
        處理單個URL，提取內容並創建WebPage對象
        
        Args:
            url: 要處理的URL
            depth: 當前深度
            parent_url: 父URL
            
        Returns:
            WebPage: 處理後的WebPage對象，如果處理失敗則返回None
        """
        try:
            # 首先檢查URL是否安全
            if self.should_skip_url(url):
                print(f"跳過不安全的URL: {url}")
                return None

            print(f"正在爬取: {url} (深度: {depth})")
            self.driver.get(url)
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception as e:
                print(f"等待頁面加載超時: {str(e)}")
                return None

            # 檢查是否有錯誤頁面標識
            if "ERR_TOO_MANY_REDIRECTS" in self.driver.page_source:
                raise WebDriverException("ERR_TOO_MANY_REDIRECTS")

            # 保存原始HTML
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            title = soup.title.string if soup.title else "No Title"
            if title == "404":
                return None

            # 轉換HTML為純文本
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            content = h.handle(str(soup))
            content = self.clean_content(content)

            # 提取麵包屑作為keywords
            keywords = [a.text for a in soup.select('.breadcrumb a')]
            
            # 使用parent_url構建path
            path = []
            if parent_url:
                parent_page = next((p for p in self.pages if p.url == parent_url), None)
                if parent_page:
                    path = parent_page.path + [parent_url]

            # 提取類別和摘要
            category = self.extract_category(url)
            summary = self.extract_summary(soup, content) 

            # 提取所有子連結
            all_links = [urljoin(self.root_url, a.get('href')) for a in soup.find_all('a', href=True)]
            child_urls = [link for link in all_links if link.startswith(self.root_url) and not self.should_skip_url(link)]

            # 創建WebPage對象
            page = WebPage(
                root_url=self.root_url,
                url=url,
                title=title,
                content=content,
                depth=depth,
                parent_url=parent_url,
                child_urls=child_urls,
                path=path,
                keywords=keywords,
                summary=summary,
                category=category,
                html=html_content
            )

            print(f"已完成爬取: {url} (深度: {depth})")
            print(f"當前進度: 已爬取 {len(self.pages)} 個頁面")
            return page

        except Exception as e:
            print(f"處理URL時出錯 {url}: {str(e)}")
            return None

    def process_url_with_retry(self, url, depth, parent_url=None, max_retries=3):
        """
        帶有重試機制的URL處理
        
        Args:
            url: 要處理的URL
            depth: 當前深度
            parent_url: 父URL
            max_retries: 最大重試次數
            
        Returns:
            WebPage: 處理後的WebPage對象，如果處理失敗則返回None
        """
        retries = 0
        while retries < max_retries:
            try:
                return self.process_url(url, depth, parent_url)
            except WebDriverException as e:
                if "ERR_TOO_MANY_REDIRECTS" in str(e) or "redirected you too many times" in str(e):
                    print(f"遇到重定向錯誤，正在重試 ({retries + 1}/{max_retries})")
                    if self.clear_cookies_and_login():
                        retries += 1
                        continue
                    
                    # 如果清理cookie和重新登入失敗，嘗試重置瀏覽器
                    self.driver = self.reset_driver()
                    if self.driver is None:
                        print("重置瀏覽器失敗，停止重試")
                        return None
                    retries += 1
                else:
                    print(f"遇到其他錯誤: {str(e)}")
                    return None
            except Exception as e:
                print(f"處理URL時發生錯誤: {str(e)}")
                return None
        
        print(f"達到最大重試次數 ({max_retries})，跳過此URL")
        return None

    def crawl_website(self):
        """
        爬取網站，使用BFS遍歷所有頁面
        
        Returns:
            List[WebPage]: 爬取的頁面列表
        """
        self.status = "Running"
        self.start_time = datetime.now().isoformat()
        
        try:
            self.driver = self.setup_driver()
            print("開始登錄...")
            if not self.login():
                print("初始登入失敗")
                self.status = "Failed: Login Error"
                return []
            print("登錄成功")

            self.visited_urls = set()
            self.pages = []
            url_queue = deque([(self.root_url, 0, None)])  # (url, depth, parent_url)
            error_count = 0
            max_errors = 5  # 最大連續錯誤次數

            while url_queue:
                current_url, depth, parent_url = url_queue.popleft()
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)            
                
                page = self.process_url_with_retry(current_url, depth, parent_url)
                
                if page:
                    error_count = 0  # 重置錯誤計數
                    self.pages.append(page)
                    
                    if depth < self.max_depth:
                        for child_url in page.child_urls:
                            if child_url not in self.visited_urls:
                                url_queue.append((child_url, depth + 1, current_url))
                else:
                    error_count += 1
                    print(f"連續錯誤次數: {error_count}")
                    if error_count >= max_errors:
                        print(f"連續錯誤次數達到上限 ({max_errors})，停止爬取")
                        break

                # 每爬取 200 個頁面後強制清理一次 cookie
                if len(self.pages) % 200 == 0 and len(self.pages) > 0:
                    print("定期清理 cookie 並重新登入")
                    self.clear_cookies_and_login()
                
                # 添加延遲，避免請求過於頻繁
                time.sleep(random.uniform(1, 3))
            
            print(f"爬取完成，共爬取了 {len(self.pages)} 個頁面")
            self.status = "Completed"
            return self.pages
            
        except Exception as e:
            print(f"爬取網站時發生錯誤: {str(e)}")
            self.status = f"Failed: {str(e)}"
            return []
        finally:
            self.end_time = datetime.now().isoformat()
            if self.driver:
                self.driver.quit()

    def save_pages_to_json(self):
        """
        將爬取的頁面保存為JSON文件
        
        Returns:
            str: 保存的文件路徑
        """
        if not self.pages:
            print("沒有頁面可保存")
            return None
            
        # 將頁面轉換為可序列化的字典
        pages_dict = []
        for page in self.pages:
            page_dict = {
                "root_url": page.root_url,
                "url": page.url,
                "title": page.title,
                "content": page.content,
                "depth": page.depth,
                "parent_url": page.parent_url,
                "child_urls": page.child_urls,
                "path": page.path,
                "keywords": page.keywords,
                "summary": page.summary,
                "category": page.category
            }
            pages_dict.append(page_dict)
            
        # 保存為JSON文件
        output_path = os.path.join(self.output_dir, "pages.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_dict, f, ensure_ascii=False, indent=2)
            
        print(f"已將頁面保存至 {output_path}")
        return output_path

    def save_sitemap(self):
        """
        創建並保存網站地圖
        
        Returns:
            str: 保存的文件路徑
        """
        if not self.pages:
            print("沒有頁面可保存")
            return None
            
        sitemap = []
        for page in self.pages:
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
            
        # 保存為JSON文件
        output_path = os.path.join(self.output_dir, "sitemap.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sitemap, f, ensure_ascii=False, indent=2)
            
        print(f"已將網站地圖保存至 {output_path}")
        return output_path

    def save_all_content(self):
        """
        將所有頁面內容保存到一個文本文件
        
        Returns:
            str: 保存的文件路徑
        """
        if not self.pages:
            print("沒有頁面可保存")
            return None
            
        output_path = os.path.join(self.output_dir, 'all_content.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in self.pages:
                f.write(f"URL: {page.url}\n")
                f.write(f"Title: {page.title}\n")
                f.write(f"Content:\n{page.content}\n")
                f.write("-" * 80 + "\n")
                
        print(f"已將所有內容保存至 {output_path}")
        return output_path

    def get_status(self):
        """
        獲取爬蟲狀態
        
        Returns:
            dict: 爬蟲狀態信息
        """
        return {
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "pages_count": len(self.pages),
            "visited_urls_count": len(self.visited_urls)
        }


# 使用示例
if __name__ == "__main__":
    # 從配置文件讀取設置
    import configparser
    config = configparser.ConfigParser()
    config.read('./config.ini')
    
    ROOT_URL = 'https://www.coolenglish.edu.tw/'
    LOGIN_URL = 'https://www.coolenglish.edu.tw/login/index.php'
    USERNAME = config['user']['username']
    PASSWORD = config['user']['password']
    
    # 創建爬蟲實例
    crawler = WebsiteCrawler(
        root_url=ROOT_URL,
        login_url=LOGIN_URL,
        username=USERNAME,
        password=PASSWORD,
        max_depth=3,  # 設置較小的深度進行測試
        output_dir=Path("./data/test_crawl")
    )
    
    # 爬取網站
    pages = crawler.crawl_website()
    
    # 保存結果
    if pages:
        crawler.save_pages_to_json()
        crawler.save_sitemap()
        crawler.save_all_content()
    
    print("爬蟲狀態:", crawler.get_status())
