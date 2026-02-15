import google.generativeai as genai
from chromadb import EmbeddingFunction, Documents, Embeddings
import config

# API設定
genai.configure(api_key=config.GOOGLE_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDBがGoogle GeminiのEmbedding APIを利用するためのラッパークラス。
    task_typeを適切に切り替えます。
    """
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.dimensionality = config.DIMENSION

    def __call__(self, input: Documents) -> Embeddings:
        """ドキュメント登録用（Ingestion時）"""
        embeddings = []
        for text in input:
            try:
                # task_type="retrieval_document" を指定
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document",
                    # output_dimensionality=self.dimensionality # 3072対応モデルの場合のみ有効化
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error embedding document: {e}")
                raise e
        return embeddings

    def embed_query(self, query: str):
        """検索クエリ用（Retrieval時）"""
        try:
            # task_type="retrieval_query" を指定
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query",
                # output_dimensionality=self.dimensionality
            )
            return result['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []