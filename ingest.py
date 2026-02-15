import os
import re
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm  # 進捗バー用ライブラリ
import config
from utils import GeminiEmbeddingFunction

def load_and_chunk_pdf(file_path):
    # プログレスバーの表示を邪魔しないよう、ここでのprintは削除し、main側で制御します
    doc = fitz.open(file_path)
    text_content = ""
    
    for page in doc:
        text_content += page.get_text() + "\n\n"
    
    # テキストクリーニング
    text_content = text_content.replace("\n\n", "<PARAGRAPH>")
    text_content = text_content.replace("\n", "")
    text_content = text_content.replace("<PARAGRAPH>", "\n\n")
    text_content = re.sub(r'[ \t]+', ' ', text_content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "。", "、", "\n", " ", ""]
    )
    
    chunks = text_splitter.create_documents([text_content])
    return chunks

def main():
    print("--- Starting Ingestion Process ---")
    
    client = chromadb.PersistentClient(path=config.DB_DIR)
    gemini_ef = GeminiEmbeddingFunction()
    
    try:
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=gemini_ef,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    pdf_files = [f for f in os.listdir(config.PDF_SOURCE_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {config.PDF_SOURCE_DIR}")
        print("Please add .pdf files to the 'data/raw_pdfs' directory.")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing...")

    # tqdmを使って進捗バーを表示
    # desc: バーの左側に表示されるタイトル
    # unit: 単位（"file"など）
    with tqdm(pdf_files, desc="Ingesting", unit="file") as pbar:
        for filename in pbar:
            # プログレスバーの右側に現在のファイル名を表示
            pbar.set_postfix(current_file=filename)
            
            file_path = os.path.join(config.PDF_SOURCE_DIR, filename)
            
            try:
                chunks = load_and_chunk_pdf(file_path)
                
                documents = [chunk.page_content for chunk in chunks]
                metadatas = [{"source": filename, "page_chunk": i} for i, chunk in enumerate(chunks)]
                ids = [f"{filename}_{i}" for i in range(len(chunks))]
                
                if documents:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    # 詳細なログを出したい場合は tqdm.write を使うとバーが崩れません
                    # tqdm.write(f"Processed {filename}: {len(chunks)} chunks added.")
            
            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")

    print(f"\nTotal documents in DB: {collection.count()}")
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    main()