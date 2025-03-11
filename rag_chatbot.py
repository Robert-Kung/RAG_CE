import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

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


class RAGChatbot:
    """RAG (Retrieval-Augmented Generation) 聊天機器人類，用於基於向量資料庫的問答"""
    
    def __init__(self, api_key: str, vectorstore_path: str = None):
        """
        初始化 RAG 聊天機器人
        
        Args:
            api_key: OpenAI API 金鑰
            vectorstore_path: 向量資料庫的路徑，如果不提供則使用預設路徑
        """
        self.api_key = api_key
        self.vectorstore_path = vectorstore_path or "./data/chroma_langchain_db"
        
        # 初始化嵌入模型
        self.embedding_model = OpenAIEmbeddings(
            model='text-embedding-3-small', 
            openai_api_key=api_key
        )
        
        # 初始化語言模型
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0, 
            api_key=api_key
        )
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048, 
            chunk_overlap=256
        )
        
        # 向量資料庫，初始為 None，需要時再載入或創建
        self.vectorstore = None
        
        # 問答鏈，初始為 None，需要時再創建
        self.qa_chain = None

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        載入現有的向量資料庫
        
        Returns:
            Chroma: 載入的向量資料庫，如果不存在則返回 None
        """
        if not os.path.exists(self.vectorstore_path):
            print(f"向量資料庫路徑不存在: {self.vectorstore_path}")
            return None
            
        try:
            vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embedding_model
            )
            self.vectorstore = vectorstore
            print(f"成功載入向量資料庫，包含 {vectorstore._collection.count()} 個文檔")
            return vectorstore
        except Exception as e:
            print(f"載入向量資料庫時出錯: {str(e)}")
            return None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        創建新的向量資料庫
        
        Args:
            documents: 要加入向量資料庫的文檔列表
            
        Returns:
            Chroma: 創建的向量資料庫
        """
        print(f"正在創建向量資料庫，共有 {len(documents)} 個文檔...")
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
        
        # 創建向量資料庫
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.vectorstore_path
        )
        
        # 持久化存儲
        vectorstore.persist()
        
        self.vectorstore = vectorstore
        print(f"向量資料庫創建完成，位於 {self.vectorstore_path}")
        return vectorstore

    def update_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        更新現有的向量資料庫，添加新文檔
        
        Args:
            documents: 要添加的新文檔列表
            
        Returns:
            Chroma: 更新後的向量資料庫
        """
        if not self.vectorstore:
            self.load_vectorstore()
            
        if not self.vectorstore:
            return self.create_vectorstore(documents)
            
        print(f"正在更新向量資料庫，添加 {len(documents)} 個新文檔...")
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
        print("向量資料庫更新完成")
        return self.vectorstore

    def process_html_to_documents(self, html_content: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        處理 HTML 內容，轉換為文檔
        
        Args:
            html_content: HTML 內容
            metadata: 文檔元數據
            
        Returns:
            List[Document]: 處理後的文檔列表
        """
        # 將 HTML 轉換為 Markdown
        markdown = MarkdownifyTransformer().transform_documents(
            [Document(page_content=html_content, metadata=metadata or {})]
        )
        
        # 分割文本
        documents = self.text_splitter.split_documents(markdown)
        return documents

    def create_qa_chain(self):
        """
        創建問答鏈
        
        Returns:
            問答鏈對象
        """
        if not self.vectorstore:
            if not self.load_vectorstore():
                raise ValueError("無法創建問答鏈，向量資料庫不存在")
        
        # 創建檢索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 創建提示模板
        template = """
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
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 創建 RAG 鏈
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.qa_chain = rag_chain
        return rag_chain

    def chat(self, query: str) -> str:
        """
        與 RAG 聊天機器人對話
        
        Args:
            query: 用戶查詢
            
        Returns:
            str: 聊天機器人的回答
        """
        if not self.qa_chain:
            self.create_qa_chain()
            
        try:
            response = self.qa_chain.invoke(query)
            return response
        except Exception as e:
            print(f"聊天過程中出錯: {str(e)}")
            return f"抱歉，處理您的問題時出現錯誤: {str(e)}"

    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        搜索與查詢相似的文檔
        
        Args:
            query: 查詢文本
            k: 返回的文檔數量
            
        Returns:
            List[Dict]: 相似文檔列表，包含內容和元數據
        """
        if not self.vectorstore:
            if not self.load_vectorstore():
                return []
                
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            
            # 格式化結果
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            return formatted_results
        except Exception as e:
            print(f"搜索相似文檔時出錯: {str(e)}")
            return []

    def convert_webpages_to_documents(self, webpages: List[Dict]) -> List[Document]:
        """
        將網頁數據轉換為文檔
        
        Args:
            webpages: 網頁數據列表
            
        Returns:
            List[Document]: 轉換後的文檔列表
        """
        documents = []
        
        for page in webpages:
            # 提取內容和元數據
            content = page.get("content", "")
            metadata = {
                "url": page.get("url", ""),
                "title": page.get("title", ""),
                "category": page.get("category", ""),
                "summary": page.get("summary", ""),
                "keywords": page.get("keywords", []),
                "depth": page.get("depth", 0)
            }
            
            # 分割文本
            page_docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )
            
            documents.extend(page_docs)
            
        print(f"已將 {len(webpages)} 個網頁轉換為 {len(documents)} 個文檔")
        return documents

    def get_vectorstore_status(self) -> Dict:
        """
        獲取向量資料庫狀態
        
        Returns:
            Dict: 向量資料庫狀態信息
        """
        exists = os.path.exists(self.vectorstore_path)
        
        if not exists:
            return {
                "status": "not_found",
                "message": "向量資料庫不存在",
                "location": self.vectorstore_path
            }
            
        try:
            # 嘗試載入向量資料庫以確認其可用性
            if not self.vectorstore:
                self.load_vectorstore()
                
            if not self.vectorstore:
                return {
                    "status": "error",
                    "message": "無法載入向量資料庫",
                    "location": self.vectorstore_path
                }
                
            # 獲取文檔數量
            doc_count = self.vectorstore._collection.count()
            
            return {
                "status": "available",
                "message": "向量資料庫可用",
                "location": self.vectorstore_path,
                "document_count": doc_count
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"檢查向量資料庫狀態時出錯: {str(e)}",
                "location": self.vectorstore_path
            }


class DocumentProcessor:
    """文檔處理類，用於處理和轉換文檔"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        初始化文檔處理器
        
        Args:
            chunk_size: 文本分割的塊大小
            chunk_overlap: 文本分割的重疊大小
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.markdownify = MarkdownifyTransformer()

    def html_to_markdown(self, html_content: str) -> str:
        """
        將 HTML 轉換為 Markdown
        
        Args:
            html_content: HTML 內容
            
        Returns:
            str: Markdown 內容
        """
        document = Document(page_content=html_content)
        markdown_docs = self.markdownify.transform_documents([document])
        return markdown_docs[0].page_content

    def split_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        分割文本為多個文檔
        
        Args:
            text: 要分割的文本
            metadata: 文檔元數據
            
        Returns:
            List[Document]: 分割後的文檔列表
        """
        return self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata] if metadata else None
        )

    def process_html_document(self, html_content: str, metadata: Dict = None) -> List[Document]:
        """
        處理 HTML 文檔，轉換為分割後的文檔列表
        
        Args:
            html_content: HTML 內容
            metadata: 文檔元數據
            
        Returns:
            List[Document]: 處理後的文檔列表
        """
        # 轉換為 Markdown
        markdown = self.html_to_markdown(html_content)
        
        # 分割文本
        documents = self.split_text(markdown, metadata)
        return documents

    def process_webpage_data(self, webpage_data: Dict) -> List[Document]:
        """
        處理網頁數據，轉換為文檔
        
        Args:
            webpage_data: 網頁數據
            
        Returns:
            List[Document]: 轉換後的文檔列表
        """
        content = webpage_data.get("content", "")
        metadata = {
            "url": webpage_data.get("url", ""),
            "title": webpage_data.get("title", ""),
            "category": webpage_data.get("category", ""),
            "summary": webpage_data.get("summary", ""),
            "keywords": webpage_data.get("keywords", []),
            "depth": webpage_data.get("depth", 0)
        }
        
        # 分割文本
        documents = self.split_text(content, metadata)
        return documents


# 使用示例
if __name__ == "__main__":
    import configparser
    
    # 讀取配置
    config = configparser.ConfigParser()
    config.read('./config.ini')
    
    OPENAI_API_KEY = config['openai']['api_key']
    VECTORSTORE_PATH = "./data/chroma_langchain_db"
    
    # 創建 RAG 聊天機器人
    chatbot = RAGChatbot(
        api_key=OPENAI_API_KEY,
        vectorstore_path=VECTORSTORE_PATH
    )
    
    # 檢查向量資料庫狀態
    status = chatbot.get_vectorstore_status()
    print("向量資料庫狀態:", status)
    
    # 如果向量資料庫存在，進行測試查詢
    if status["status"] == "available":
        query = "什麼是 Cool English？"
        print(f"\n問題: {query}")
        answer = chatbot.chat(query)
        print(f"回答: {answer}")
        
        # 搜索相似文檔
        print("\n搜索相似文檔:")
        docs = chatbot.search_similar_documents(query, k=2)
        for i, doc in enumerate(docs):
            print(f"\n文檔 {i+1}:")
            print(f"內容: {doc['content'][:100]}...")
            print(f"元數據: {doc['metadata']}")
