import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3


class DatabaseManager:
    """資料庫管理類，用於處理資料的持久化儲存和檢索"""
    
    def __init__(self, db_path: str = "./data/database.db"):
        """
        初始化資料庫管理器
        
        Args:
            db_path: 資料庫文件路徑
        """
        self.db_path = db_path
        
        # 確保資料庫目錄存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化資料庫
        self.init_db()
    
    def init_db(self):
        """初始化資料庫結構"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建網頁資料表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS webpages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            html TEXT,
            summary TEXT,
            category TEXT,
            depth INTEGER,
            parent_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 創建關鍵字資料表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webpage_id INTEGER,
            keyword TEXT,
            FOREIGN KEY (webpage_id) REFERENCES webpages (id)
        )
        ''')
        
        # 創建子連結資料表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS child_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webpage_id INTEGER,
            child_url TEXT,
            FOREIGN KEY (webpage_id) REFERENCES webpages (id)
        )
        ''')
        
        # 創建查詢歷史資料表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_webpage(self, webpage: Dict) -> int:
        """
        儲存網頁資料
        
        Args:
            webpage: 網頁資料字典
            
        Returns:
            int: 插入的記錄 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 插入網頁資料
            cursor.execute('''
            INSERT OR REPLACE INTO webpages (url, title, content, html, summary, category, depth, parent_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                webpage.get('url', ''),
                webpage.get('title', ''),
                webpage.get('content', ''),
                webpage.get('html', ''),
                webpage.get('summary', ''),
                webpage.get('category', ''),
                webpage.get('depth', 0),
                webpage.get('parent_url', None)
            ))
            
            webpage_id = cursor.lastrowid
            
            # 插入關鍵字
            keywords = webpage.get('keywords', [])
            for keyword in keywords:
                cursor.execute('''
                INSERT INTO keywords (webpage_id, keyword)
                VALUES (?, ?)
                ''', (webpage_id, keyword))
            
            # 插入子連結
            child_urls = webpage.get('child_urls', [])
            for child_url in child_urls:
                cursor.execute('''
                INSERT INTO child_urls (webpage_id, child_url)
                VALUES (?, ?)
                ''', (webpage_id, child_url))
            
            conn.commit()
            return webpage_id
        except Exception as e:
            conn.rollback()
            print(f"儲存網頁資料時出錯: {str(e)}")
            return -1
        finally:
            conn.close()
    
    def store_webpages(self, webpages: List[Dict]) -> int:
        """
        批量儲存網頁資料
        
        Args:
            webpages: 網頁資料列表
            
        Returns:
            int: 成功插入的記錄數
        """
        success_count = 0
        for webpage in webpages:
            if self.store_webpage(webpage) > 0:
                success_count += 1
        return success_count
    
    def get_webpage(self, url: str) -> Optional[Dict]:
        """
        根據 URL 獲取網頁資料
        
        Args:
            url: 網頁 URL
            
        Returns:
            Dict: 網頁資料，如果不存在則返回 None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 獲取網頁基本資料
            cursor.execute('''
            SELECT * FROM webpages WHERE url = ?
            ''', (url,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            webpage = dict(row)
            webpage_id = webpage['id']
            
            # 獲取關鍵字
            cursor.execute('''
            SELECT keyword FROM keywords WHERE webpage_id = ?
            ''', (webpage_id,))
            
            keywords = [row['keyword'] for row in cursor.fetchall()]
            webpage['keywords'] = keywords
            
            # 獲取子連結
            cursor.execute('''
            SELECT child_url FROM child_urls WHERE webpage_id = ?
            ''', (webpage_id,))
            
            child_urls = [row['child_url'] for row in cursor.fetchall()]
            webpage['child_urls'] = child_urls
            
            return webpage
        except Exception as e:
            print(f"獲取網頁資料時出錯: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_all_webpages(self) -> List[Dict]:
        """
        獲取所有網頁資料
        
        Returns:
            List[Dict]: 網頁資料列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 獲取所有網頁 ID
            cursor.execute('''
            SELECT id FROM webpages
            ''')
            
            webpage_ids = [row['id'] for row in cursor.fetchall()]
            webpages = []
            
            # 獲取每個網頁的詳細資料
            for webpage_id in webpage_ids:
                cursor.execute('''
                SELECT * FROM webpages WHERE id = ?
                ''', (webpage_id,))
                
                webpage = dict(cursor.fetchone())
                
                # 獲取關鍵字
                cursor.execute('''
                SELECT keyword FROM keywords WHERE webpage_id = ?
                ''', (webpage_id,))
                
                keywords = [row['keyword'] for row in cursor.fetchall()]
                webpage['keywords'] = keywords
                
                # 獲取子連結
                cursor.execute('''
                SELECT child_url FROM child_urls WHERE webpage_id = ?
                ''', (webpage_id,))
                
                child_urls = [row['child_url'] for row in cursor.fetchall()]
                webpage['child_urls'] = child_urls
                
                webpages.append(webpage)
            
            return webpages
        except Exception as e:
            print(f"獲取所有網頁資料時出錯: {str(e)}")
            return []
        finally:
            conn.close()
    
    def store_query(self, query: str, answer: str) -> int:
        """
        儲存查詢歷史
        
        Args:
            query: 用戶查詢
            answer: 系統回答
            
        Returns:
            int: 插入的記錄 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO query_history (query, answer)
            VALUES (?, ?)
            ''', (query, answer))
            
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            print(f"儲存查詢歷史時出錯: {str(e)}")
            return -1
        finally:
            conn.close()
    
    def get_query_history(self, limit: int = 100) -> List[Dict]:
        """
        獲取查詢歷史
        
        Args:
            limit: 返回的記錄數量限制
            
        Returns:
            List[Dict]: 查詢歷史列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT * FROM query_history
            ORDER BY created_at DESC
            LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"獲取查詢歷史時出錯: {str(e)}")
            return []
        finally:
            conn.close()
    
    def export_webpages_to_json(self, output_path: str) -> bool:
        """
        將網頁資料匯出為 JSON 文件
        
        Args:
            output_path: 輸出文件路徑
            
        Returns:
            bool: 是否成功匯出
        """
        try:
            webpages = self.get_all_webpages()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(webpages, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"匯出網頁資料時出錯: {str(e)}")
            return False
    
    def import_webpages_from_json(self, input_path: str) -> int:
        """
        從 JSON 文件導入網頁資料
        
        Args:
            input_path: 輸入文件路徑
            
        Returns:
            int: 成功導入的記錄數
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                webpages = json.load(f)
                
            return self.store_webpages(webpages)
        except Exception as e:
            print(f"導入網頁資料時出錯: {str(e)}")
            return 0


# 使用示例
if __name__ == "__main__":
    db_manager = DatabaseManager(db_path="./data/test_db.db")
    
    # 測試儲存網頁
    webpage = {
        "url": "https://www.example.com",
        "title": "Example Page",
        "content": "This is an example page content.",
        "html": "<html><body>Example</body></html>",
        "summary": "Example summary",
        "category": "Test",
        "depth": 1,
        "parent_url": None,
        "keywords": ["example", "test"],
        "child_urls": ["https://www.example.com/child1", "https://www.example.com/child2"]
    }
    
    webpage_id = db_manager.store_webpage(webpage)
    print(f"儲存網頁，ID: {webpage_id}")
    
    # 測試獲取網頁
    retrieved_webpage = db_manager.get_webpage("https://www.example.com")
    if retrieved_webpage:
        print(f"獲取網頁: {retrieved_webpage['title']}")
    
    # 測試儲存查詢
    query_id = db_manager.store_query("這是什麼網站？", "這是一個範例網站。")
    print(f"儲存查詢，ID: {query_id}")
    
    # 測試獲取查詢歷史
    query_history = db_manager.get_query_history()
    print(f"查詢歷史數量: {len(query_history)}")
    
    # 測試匯出網頁資料
    if db_manager.export_webpages_to_json("./data/export_test.json"):
        print("成功匯出網頁資料")
