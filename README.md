# Research AI Assistant (Gemini-powered RAG)
arXivから論文を収集し、Google Gemini APIを使用して専門的な回答を行う研究支援ツールです。

## 主な機能
- **arXiv連携**: キーワードから最新論文を自動ダウンロード (`save_pdf_from_arXiv.py`)
- **高速検索 (RAG)**: ChromaDBとGemini Embeddingを使用して関連箇所を特定 (`ingest.py`, `utils.py`)
- **Web UI**: Streamlitによる直感的なチャットインターフェース (`streamlit_app.py`)
- **CLIモード**: ターミナルからの対話も可能 (`app.py`)

## セットアップ
1. リポジトリをクローン
2. `.env` ファイルを作成し `GOOGLE_API_KEY=あなたのキー` を設定
3. `pip install -r requirements.txt` でライブラリをインストール
4. `python ingest.py` でPDFをベクトル化
5. `streamlit run streamlit_app.py` で起動