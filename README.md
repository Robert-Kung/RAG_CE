# Cool English RAG 聊天機器人

這是一個基於 RAG (Retrieval-Augmented Generation) 技術的聊天機器人系統，專門為 Cool English 網站設計。它可以爬取網站內容，建立向量資料庫，並根據使用者的問題提供相關答案。

## 系統架構

系統由以下幾個主要模組組成：

1. **crawler.py**: 網站爬蟲模組，負責爬取 Cool English 網站的內容。
2. **rag_chatbot.py**: RAG 聊天機器人模組，負責處理使用者的問題並生成答案。
3. **db.py**: 資料庫管理模組，負責處理資料的持久化儲存和檢索。
4. **api.py**: API 服務模組，提供 RESTful API 接口。
5. **main.py**: 為舊程式位置，需要加以改寫成主程式入口點，目前暫定由api.py進入。

## 環境設置

### 前置條件

- Docker 和 Docker Compose
- OpenAI API 金鑰
- Cool English 網站的帳號和密碼

### 配置文件

在專案根目錄創建 `.env` 文件，內容如下：

```
OPENAI_API_KEY=your_openai_api_key
WEBSITE_USERNAME=your_coolenglish_username
WEBSITE_PASSWORD=your_coolenglish_password
```

## 安裝與運行

### 使用 Docker Compose

1. 克隆專案：

```bash
git clone https://github.com/Robert-Kung/RAG_CE.git
cd RAG_CE
```

2. 創建 `.env` 文件（如上所述）。

3. 構建並啟動容器：

```bash
docker-compose up -d
```

4. 查看日誌：

```bash
docker-compose logs -f
```

## 系統使用流程

建議使用 Postman 或類似的 API 測試工具來測試系統功能。以下是基本使用流程：

### 1. 爬取網站內容

首先，啟動爬蟲來收集 Cool English 網站的內容。可以設定爬取深度，預設為 20。

### 2. 創建向量資料庫

爬蟲完成後，創建向量資料庫以便進行語義搜索。

### 3. 使用聊天機器人

向量資料庫建立完成後，即可開始使用聊天機器人進行問答。

### 4. 其他操作

系統還提供了搜索相似文檔、管理網頁資料等功能，詳細請參考 API 文檔。

## API 文檔

### 爬蟲相關 API

- `POST /api/crawler/start`: 啟動爬蟲
  - 參數：`max_depth` (可選，預設為 20)
  - 範例：`{"max_depth": 20}`

- `GET /api/crawler/status`: 獲取爬蟲狀態
  - 回傳：爬蟲當前狀態，包括已爬取的頁面數量和狀態

### 聊天機器人相關 API

- `POST /api/chat`: 發送查詢請求
  - 參數：`query` (必填)
  - 範例：`{"query": "什麼是 Cool English？"}`

- `POST /api/chat/stream`: 使用流式回應
  - 參數：`query` (必填)
  - 範例：`{"query": "什麼是 Cool English？"}`

- `POST /api/search`: 搜索相似文檔
  - 參數：`query` (必填)，`k` (可選，預設為 4)
  - 範例：`{"query": "Cool English 的課程", "k": 5}`

### 向量資料庫相關 API

- `GET /api/vectorstore/status`: 獲取向量資料庫狀態
  - 回傳：向量資料庫狀態，包括文檔數量和狀態

- `POST /api/vectorstore/create`: 創建向量資料庫
  - 參數：`file_path` (可選，預設為 "./data/page_data/pages.json")
  - 範例：`{"file_path": "./data/page_data/pages.json"}`

- `POST /api/vectorstore/update`: 更新向量資料庫
  - 參數：`file_path` (可選，預設為 "./data/page_data/pages.json")
  - 範例：`{"file_path": "./data/page_data/pages.json"}`

### 資料庫相關 API

- `GET /api/db/webpages`: 獲取所有網頁資料
  - 參數：`page` (可選，預設為 1)，`per_page` (可選，預設為 20)
  - 範例：`/api/db/webpages?page=1&per_page=20`

- `GET /api/db/webpage`: 獲取特定網頁資料
  - 參數：`url` (必填)
  - 範例：`/api/db/webpage?url=https://www.coolenglish.edu.tw/course/view.php?id=123`

- `GET /api/db/query_history`: 獲取查詢歷史
  - 參數：`limit` (可選，預設為 100)
  - 範例：`/api/db/query_history?limit=50`

- `POST /api/db/export`: 匯出網頁資料
  - 參數：`output_path` (可選，預設為 "./data/export_webpages.json")
  - 範例：`{"output_path": "./data/export_webpages.json"}`

- `POST /api/db/import`: 導入網頁資料
  - 參數：`input_path` (必填)
  - 範例：`{"input_path": "./data/export_webpages.json"}`

## 系統架構圖

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Website Crawler |---->| Vector Database  |---->|   RAG Chatbot    |
|  (crawler.py)    |     | (rag_chatbot.py) |     | (rag_chatbot.py) |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
| Database Manager |<--->|    API Server    |<--->|     Client       |
|     (db.py)      |     |    (api.py)      |     |  (Web/Mobile)    |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```

## 目錄結構

```
.
├── api.py                 # API 服務模組
├── crawler.py             # 網站爬蟲模組
├── db.py                  # 資料庫管理模組
├── docker-compose.yml     # Docker Compose 配置
├── Dockerfile             # Docker 配置
├── main.py                # 主程式入口點
├── rag_chatbot.py         # RAG 聊天機器人模組
├── requirements.txt       # Python 依賴
├── .env                   # 環境變數配置
└── data/                  # 資料目錄
    ├── page_data/         # 爬取的網頁資料
    └── chroma_langchain_db/ # 向量資料庫
```

## 故障排除

### 爬蟲無法登入

檢查您的帳號和密碼是否正確，並確保 `.env` 文件已正確設置。

### 向量資料庫創建失敗

確保爬蟲已成功完成，並且 `./data/page_data/pages.json` 文件存在且包含有效的數據。

### API 服務無法啟動

檢查日誌以獲取更多信息：

```bash
docker-compose logs -f
```

## 注意事項

1. 首次運行時，爬蟲可能需要較長時間才能完成，視網站內容量而定。
2. 向量資料庫的創建也可能需要一些時間，特別是當爬取的頁面數量較多時。
3. 請確保您有足夠的磁盤空間來存儲爬取的數據和向量資料庫。

