import streamlit as st
import google.generativeai as genai
import chromadb
import os
import shutil
import config
from utils import GeminiEmbeddingFunction
# ingest.py からPDF処理関数を読み込む
from ingest import load_and_chunk_pdf

# --- 設定と初期化 ---
st.set_page_config(page_title="Research AI Assistant", page_icon="", layout="wide")

# API設定
genai.configure(api_key=config.GOOGLE_API_KEY)

# --- リソースのキャッシュ化 ---
@st.cache_resource
def load_db_and_model():
    try:
        gemini_ef = GeminiEmbeddingFunction()
        client = chromadb.PersistentClient(path=config.DB_DIR)
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME, 
            embedding_function=gemini_ef
        )
        model = genai.GenerativeModel(config.GENERATION_MODEL)
        return collection, model, gemini_ef
    except Exception as e:
        st.error(f"DBエラー: {e}")
        return None, None, None

collection, model, gemini_ef = load_db_and_model()

# --- サイドバー：文献管理機能 ---
with st.sidebar:
    st.markdown("---")
    show_debug = st.checkbox("デバックモードを表示", value=False)

    st.header("文献管理")
    
    # 1. 新規PDF追加
    st.subheader("新規PDFの追加")
    uploaded_files = st.file_uploader("PDFをアップロード", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("追加して学習開始"):
        if collection:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"処理中: {uploaded_file.name}...")
                
                # A. ファイルを保存
                save_path = os.path.join(config.PDF_SOURCE_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # B. テキスト抽出とチャンク化 (ingest.pyの関数を利用)
                try:
                    chunks = load_and_chunk_pdf(save_path)
                    
                    # C. DBへの登録
                    documents = [chunk.page_content for chunk in chunks]
                    metadatas = [{"source": uploaded_file.name, "page_chunk": idx} for idx, chunk in enumerate(chunks)]
                    ids = [f"{uploaded_file.name}_{idx}" for idx in range(len(chunks))]
                    
                    if documents:
                        collection.add(documents=documents, metadatas=metadatas, ids=ids)
                        st.success(f"✅ {uploaded_file.name} をデータベースに追加しました")
                    
                except Exception as e:
                    st.error(f"エラー ({uploaded_file.name}): {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("すべての処理が完了しました！")
            st.rerun() # 画面をリロードしてリストを更新

    st.markdown("---")

    # 2. 登録済み文献リスト表示
    st.subheader("登録済み文献一覧")
    if collection:
        # DBから全メタデータを取得してファイル名のみ抽出
        # (件数が多い場合は重くなる可能性があるため、上限を設けるかページネーションが必要ですが、個人利用ならこれでOK)
        try:
            all_data = collection.get(include=["metadatas"])
            unique_sources = set()
            for meta in all_data["metadatas"]:
                if meta and "source" in meta:
                    unique_sources.add(meta["source"])
            
            if unique_sources:
                for source in sorted(list(unique_sources)):
                    st.markdown(f"-  {source}")
                st.caption(f"合計: {len(unique_sources)} ファイル")
            else:
                st.info("まだ文献が登録されていません。")
        except Exception as e:
            st.error("リストの取得に失敗しました")

# --- メインエリア：チャット機能 ---
st.title("🤖 Research AI Assistant")
st.caption(f"Powered by {config.GENERATION_MODEL}")

# 履歴管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("参照文献"):
                st.write(message["sources"])

# 入力フォーム
st.markdown("---")
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("質問を入力:", height=100, placeholder="質問を入力...\n(Ctrl+Enter で送信)")
    col1, col2 = st.columns([1, 6])
    with col1:
        submit_button = st.form_submit_button("送信", type="primary")

# 処理ロジック
if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    if collection and model:
        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            
            with st.spinner("文献を検索中..."):
                query_vector = gemini_ef.embed_query(prompt)
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=20, # 文献を多めに取得
                    include=["documents", "metadatas", "distances"]
                )
            
            if show_debug:
                with st.expander(" デバッグ: 検索された生データ (Embedding確認)", expanded=True):
                    st.write(f"検索クエリ: {prompt}")
                    
                    if results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            score = results['distances'][0][i]
                            meta = results['metadatas'][0][i]
                            src = meta.get('source', 'Unknown')
                            
                            # スコアと内容を表示
                            st.markdown(f"**Rank {i+1}** |  `{src}` |  Distance: `{score:.4f}`")
                            st.text(doc[:150] + "...") # 長いので先頭だけ表示
                            st.divider()
                    else:
                        st.warning("検索結果が0件です。エンベディングがうまくいっていない可能性があります。")

            context_text = ""
            sources = set()
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    src = meta.get('source', 'Unknown')
                    sources.add(src)
                    context_text += f"<doc source='{src}'>\n{doc}\n</doc>\n\n"
            
            system_prompt = f"""
            あなたは優秀な研究アシスタントです。
            以下の「検索された文献データ」を使用して、ユーザーの質問に日本語で回答してください。
            
            【重要事項】
            - データはPDFから自動抽出されたもので、誤字やレイアウト崩れが含まれる可能性があります。
            - 複数の文献に情報がある場合は統合して答えてください。
            - 文献に情報がない場合は、正直に「情報が見つかりません」と答えてください。

            【文献データ】
            {context_text}

            【質問】
            {prompt}
            """

            try:
                full_response = ""
                response = model.generate_content(system_prompt, stream=True)
                
                for chunk in response:
                    full_response += chunk.text
                    msg_placeholder.markdown(full_response + "▌")
                
                msg_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("今回の回答に使用した文献"):
                        st.write(list(sources))

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": list(sources)
                })

            except Exception as e:
                st.error(f"生成エラー: {e}")