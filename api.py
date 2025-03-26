import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import configparser
import threading
import time

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# 導入其他模組
# 假設這些是從其他檔案導入的
from crawler import WebsiteCrawler, WebPage
from rag_chatbot import RAGChatbot, DocumentProcessor
from db import DatabaseManager
from dotenv import load_dotenv

# 配置讀取
config = configparser.ConfigParser()
config.read('./config.ini')
load_dotenv()
# 環境變數
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

ROOT_URL = 'https://www.coolenglish.edu.tw/'
LOGIN_URL = 'https://www.coolenglish.edu.tw/login/index.php'
USERNAME = os.environ.get('WEBSITE_USERNAME') or config['user']['username']
PASSWORD = os.environ.get('WEBSITE_PASSWORD') or config['user']['password']
VECTORSTORE_PATH = "./data/chroma_langchain_db"
OUTPUT_DIR = "./data/page_data"

# 全局變數
crawler_status = {"status": "Not Started", "progress": 0, "total": 0}
chatbot = None
crawler = None


class APIServer:
    """API 服務器類，用於提供 Web API 服務"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """
        初始化 API 服務器
        
        Args:
            host: 主機地址
            port: 端口號
        """
        self.app = Flask(__name__)
        CORS(self.app)  # 啟用 CORS
        self.host = host
        self.port = port
        
        # 初始化全局變數
        self.init_globals()
        
        # 註冊路由
        self.register_routes()
        
    def init_globals(self):
        """初始化全局變數"""
        global chatbot, crawler
        
        # 初始化 RAG 聊天機器人
        chatbot = RAGChatbot(
            api_key=OPENAI_API_KEY,
            vectorstore_path=VECTORSTORE_PATH
        )
        
        # 初始化爬蟲
        crawler = WebsiteCrawler(
            root_url=ROOT_URL,
            login_url=LOGIN_URL,
            username=USERNAME,
            password=PASSWORD,
            max_depth=20,
            output_dir=Path(OUTPUT_DIR)
        )

        # 初始化資料庫管理器
        db_manager = DatabaseManager()
        
    def register_routes(self):
        """註冊 API 路由"""
        # 首頁
        @self.app.route('/')
        def home():
            return jsonify({
                "status": "ok",
                "message": "Cool English RAG API 服務運行中",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            })
        
        # 健康檢查
        @self.app.route('/health')
        def health_check():
            return jsonify({
                "status": "ok",
                "services": {
                    "api": "running",
                    "chatbot": "available" if chatbot else "not_initialized",
                    "crawler": "available" if crawler else "not_initialized"
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # 爬蟲相關 API
        @self.app.route('/api/crawler/start', methods=['POST'])
        def start_crawler():
            return self.handle_start_crawler()
            
        @self.app.route('/api/crawler/status')
        def get_crawler_status():
            return self.handle_get_crawler_status()
        
        # 聊天機器人相關 API
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            return self.handle_chat()
            
        @self.app.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            return self.handle_chat_stream()
            
        @self.app.route('/api/search', methods=['POST'])
        def search():
            return self.handle_search()
            
        @self.app.route('/api/vectorstore/status')
        def vectorstore_status():
            return self.handle_vectorstore_status()
            
        @self.app.route('/api/vectorstore/create', methods=['POST'])
        def create_vectorstore():
            return self.handle_create_vectorstore()
            
        @self.app.route('/api/vectorstore/update', methods=['POST'])
        def update_vectorstore():
            return self.handle_update_vectorstore()
        
        # 資料庫相關 API
        @self.app.route('/api/db/webpages', methods=['GET'])
        def get_all_webpages():
            return self.handle_get_all_webpages()

        @self.app.route('/api/db/webpage', methods=['GET'])
        def get_webpage():
            return self.handle_get_webpage()

        @self.app.route('/api/db/query_history', methods=['GET'])
        def get_query_history():
            return self.handle_get_query_history()

        @self.app.route('/api/db/export', methods=['POST'])
        def export_webpages():
            return self.handle_export_webpages()

        @self.app.route('/api/db/import', methods=['POST'])
        def import_webpages():
            return self.handle_import_webpages()

    def handle_start_crawler(self) -> Response:
        """
        處理啟動爬蟲的請求
        
        Returns:
            Response: Flask 回應
        """
        global crawler_status
        
        # 檢查爬蟲是否已在運行
        if crawler_status["status"] == "Running":
            return jsonify({
                "status": "error",
                "message": "爬蟲已在運行中"
            }), 400
        
        # 獲取請求參數
        data = request.json or {}
        max_depth = data.get('max_depth', 20)
        
        # 更新爬蟲配置
        crawler.max_depth = max_depth
        
        # 在背景執行爬蟲
        thread = threading.Thread(target=self.run_crawler_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "爬蟲已啟動",
            "max_depth": max_depth
        })
    
    def run_crawler_in_background(self):
        """在背景執行爬蟲"""
        global crawler_status
        
        crawler_status = {"status": "Running", "progress": 0, "total": 0}
        
        try:
            # 執行爬蟲
            pages = crawler.crawl_website()
            
            # 保存結果
            if pages:
                crawler.save_pages_to_json()
                crawler.save_sitemap()
                crawler.save_all_content()
                
                # 更新狀態
                crawler_status = {
                    "status": "Completed",
                    "progress": len(pages),
                    "total": len(pages),
                    "message": f"爬蟲完成，共爬取了 {len(pages)} 個頁面",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                crawler_status = {
                    "status": "Failed",
                    "message": "爬蟲失敗，未爬取到任何頁面",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            crawler_status = {
                "status": "Error",
                "message": f"爬蟲過程中發生錯誤: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def handle_get_crawler_status(self) -> Response:
        """
        處理獲取爬蟲狀態的請求
        
        Returns:
            Response: Flask 回應
        """
        global crawler_status
        
        # 如果爬蟲已初始化，獲取其狀態
        if crawler:
            crawler_info = crawler.get_status()
            crawler_status.update({
                "pages_count": crawler_info.get("pages_count", 0),
                "visited_urls_count": crawler_info.get("visited_urls_count", 0)
            })
        
        return jsonify(crawler_status)
    
    def handle_chat(self) -> Response:
        """
        處理聊天請求
        
        Returns:
            Response: Flask 回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        # 獲取請求參數
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 'query'"
            }), 400
        
        query = data['query']
        
        try:
            # 獲取回答
            answer = chatbot.chat(query)
            
            # 記錄查詢歷史
            db_manager.store_query(query, answer)
            
            return jsonify({
                "status": "success",
                "query": query,
                "answer": answer
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"處理聊天請求時出錯: {str(e)}"
            }), 500

    def handle_chat_stream(self) -> Response:
        """
        處理流式聊天請求
        
        Returns:
            Response: Flask 流式回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        # 獲取請求參數
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 'query'"
            }), 400
        
        query = data['query']
        
        # 模擬流式回應（實際實現需要使用支持流式輸出的 LLM）
        def generate():
            try:
                # 獲取回答
                answer = chatbot.chat(query)
                
                # 模擬流式輸出
                words = answer.split()
                for i in range(0, len(words), 3):
                    chunk = ' '.join(words[i:i+3])
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    time.sleep(0.1)
                
                # 發送完成信號
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(stream_with_context(generate()), 
                       mimetype='text/event-stream')
    
    def handle_search(self) -> Response:
        """
        處理搜索請求
        
        Returns:
            Response: Flask 回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        # 獲取請求參數
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 'query'"
            }), 400
        
        query = data['query']
        k = data.get('k', 5)  # 默認返回 5 個結果
        
        try:
            # 搜索相似文檔
            results = chatbot.search_similar_documents(query, k=k)
            
            return jsonify({
                "status": "success",
                "query": query,
                "results_count": len(results),
                "results": results
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"處理搜索請求時出錯: {str(e)}"
            }), 500
    
    def handle_vectorstore_status(self) -> Response:
        """
        處理獲取向量資料庫狀態的請求
        
        Returns:
            Response: Flask 回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        try:
            # 獲取向量資料庫狀態
            status = chatbot.get_vectorstore_status()
            
            return jsonify({
                "status": "success",
                "vectorstore": status
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"獲取向量資料庫狀態時出錯: {str(e)}"
            }), 500
    
    def handle_create_vectorstore(self) -> Response:
        """
        處理創建向量資料庫的請求
        
        Returns:
            Response: Flask 回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        try:
            # 檢查是否存在爬取的頁面數據
            pages_path = os.path.join(OUTPUT_DIR, "pages.json")
            if not os.path.exists(pages_path):
                return jsonify({
                    "status": "error",
                    "message": "找不到爬取的頁面數據，請先執行爬蟲"
                }), 400
            
            # 載入頁面數據
            with open(pages_path, 'r', encoding='utf-8') as f:
                pages = json.load(f)
            
            # 創建文檔處理器
            processor = DocumentProcessor()
            
            # 處理頁面數據
            documents = []
            for page in pages:
                page_docs = processor.process_webpage_data(page)
                documents.extend(page_docs)
            
            # 創建向量資料庫
            if documents:
                chatbot.create_vectorstore(documents)
                
                return jsonify({
                    "status": "success",
                    "message": f"成功創建向量資料庫，包含 {len(documents)} 個文檔",
                    "document_count": len(documents)
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "沒有可處理的文檔"
                }), 400
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"創建向量資料庫時出錯: {str(e)}"
            }), 500
    
    def handle_update_vectorstore(self) -> Response:
        """
        處理更新向量資料庫的請求
        
        Returns:
            Response: Flask 回應
        """
        # 檢查聊天機器人是否已初始化
        if not chatbot:
            return jsonify({
                "status": "error",
                "message": "聊天機器人未初始化"
            }), 500
        
        try:
            # 獲取請求參數
            data = request.json or {}
            file_path = data.get('file_path')
            
            if not file_path:
                return jsonify({
                    "status": "error",
                    "message": "缺少必要參數 'file_path'"
                }), 400
            
            # 檢查文件是否存在
            if not os.path.exists(file_path):
                return jsonify({
                    "status": "error",
                    "message": f"找不到指定文件: {file_path}"
                }), 400
            
            # 載入數據
            with open(file_path, 'r', encoding='utf-8') as f:
                pages = json.load(f)
            
            # 創建文檔處理器
            processor = DocumentProcessor()
            
            # 處理頁面數據
            documents = []
            for page in pages:
                page_docs = processor.process_webpage_data(page)
                documents.extend(page_docs)
            
            # 更新向量資料庫
            if documents:
                chatbot.update_vectorstore(documents)
                
                return jsonify({
                    "status": "success",
                    "message": f"成功更新向量資料庫，添加 {len(documents)} 個文檔",
                    "document_count": len(documents)
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "沒有可處理的文檔"
                }), 400
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"更新向量資料庫時出錯: {str(e)}"
            }), 500
    
    def handle_get_all_webpages(self) -> Response:
        """
        處理獲取所有網頁資料的請求
        
        Returns:
            Response: Flask 回應
        """
        try:
            # 獲取分頁參數
            page = request.args.get('page', default=1, type=int)
            per_page = request.args.get('per_page', default=20, type=int)
            
            # 獲取所有網頁資料
            webpages = db_manager.get_all_webpages()
            
            # 計算分頁
            start = (page - 1) * per_page
            end = start + per_page
            paginated_webpages = webpages[start:end]
            
            return jsonify({
                "status": "success",
                "total": len(webpages),
                "page": page,
                "per_page": per_page,
                "webpages": paginated_webpages
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"獲取網頁資料時出錯: {str(e)}"
            }), 500

    def handle_get_webpage(self) -> Response:
        """
        處理獲取單個網頁資料的請求
        
        Returns:
            Response: Flask 回應
        """
        # 獲取 URL 參數
        url = request.args.get('url')
        if not url:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 'url'"
            }), 400
        
        try:
            # 獲取網頁資料
            webpage = db_manager.get_webpage(url)
            
            if webpage:
                return jsonify({
                    "status": "success",
                    "webpage": webpage
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"找不到 URL 為 {url} 的網頁"
                }), 404
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"獲取網頁資料時出錯: {str(e)}"
            }), 500

    def handle_get_query_history(self) -> Response:
        """
        處理獲取查詢歷史的請求
        
        Returns:
            Response: Flask 回應
        """
        try:
            # 獲取分頁參數
            limit = request.args.get('limit', default=100, type=int)
            
            # 獲取查詢歷史
            query_history = db_manager.get_query_history(limit=limit)
            
            return jsonify({
                "status": "success",
                "total": len(query_history),
                "limit": limit,
                "query_history": query_history
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"獲取查詢歷史時出錯: {str(e)}"
            }), 500

    def handle_export_webpages(self) -> Response:
        """
        處理匯出網頁資料的請求
        
        Returns:
            Response: Flask 回應
        """
        # 獲取請求參數
        data = request.json or {}
        output_path = data.get('output_path', './data/export_webpages.json')
        
        try:
            # 匯出網頁資料
            success = db_manager.export_webpages_to_json(output_path)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"成功匯出網頁資料至 {output_path}"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "匯出網頁資料失敗"
                }), 500
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"匯出網頁資料時出錯: {str(e)}"
            }), 500

    def handle_import_webpages(self) -> Response:
        """
        處理導入網頁資料的請求
        
        Returns:
            Response: Flask 回應
        """
        # 獲取請求參數
        data = request.json or {}
        input_path = data.get('input_path')
        
        if not input_path:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 'input_path'"
            }), 400
        
        try:
            # 導入網頁資料
            count = db_manager.import_webpages_from_json(input_path)
            
            return jsonify({
                "status": "success",
                "message": f"成功導入 {count} 筆網頁資料",
                "count": count
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"導入網頁資料時出錯: {str(e)}"
            }), 500
        
    def run(self):
        """啟動 API 服務器"""
        self.app.run(host=self.host, port=self.port, debug=False)


# 使用示例
if __name__ == "__main__":
    server = APIServer(host='0.0.0.0', port=5000)
    server.run()
