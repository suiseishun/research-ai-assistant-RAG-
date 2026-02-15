import google.generativeai as genai
import chromadb
import config
from utils import GeminiEmbeddingFunction
import sys

def main():
    # 1. DBとモデルの準備
    print("Loading Knowledge Base...")
    client = chromadb.PersistentClient(path=config.DB_DIR)
    gemini_ef = GeminiEmbeddingFunction()
    
    try:
        collection = client.get_collection(
            name=config.COLLECTION_NAME, 
            embedding_function=gemini_ef
        )
    except Exception as e:
        print(f"Error: Could not load collection. Have you run 'ingest.py'?\nDetail: {e}")
        return

    # 生成モデルの準備
    model = genai.GenerativeModel(config.GENERATION_MODEL)
    
    print("\n" + "="*50)
    print(f" Research AI Assistant (Gemini-powered)")
    print(f" Model: {config.GENERATION_MODEL}")
    print(" Type 'exit' to quit.")
    print("="*50 + "\n")

    # 2. チャットループ
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "終了"]:
            break
        if not user_query.strip():
            continue

        print("Searching relevant papers...")
        
        # A. クエリのベクトル化
        query_vector = gemini_ef.embed_query(user_query)
        
        # B. 検索実行 (Top 5)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=20,
            include=["documents", "metadatas"]
        )
        
        # C. コンテキストの構築
        context_text = ""
        sources = set()
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                source_file = meta.get('source', 'Unknown')
                sources.add(source_file)
                context_text += f"--- Source: {source_file} ---\n{doc}\n\n"
        else:
            print("No relevant documents found.")
            continue

        # D. プロンプト作成と回答生成
        prompt = f"""
        あなたは専門的な研究アシスタントです。
        以下の「検索された論文の抜粋」に基づいて、ユーザーの質問に日本語で論理的に答えてください。
        
        【検索された論文の抜粋】
        {context_text}

        【ユーザーの質問】
        {user_query}
        """

        print("Thinking...")
        try:
            response = model.generate_content(prompt)
            print(f"\nAI: {response.text}")
            print("\n[Reference Sources]:", ", ".join(sources))
            print("-" * 50)
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()