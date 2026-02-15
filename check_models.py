import google.generativeai as genai
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# APIキーの設定
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ エラー: .env ファイルが見つからないか、GOOGLE_API_KEY が設定されていません。")
    exit()

genai.configure(api_key=api_key)

print("========================================")
print("🤖 利用可能な Google Gemini モデル一覧")
print("========================================")

try:
    print("\n--- 📝 生成モデル (チャット/文章作成用) ---")
    # generateContent メソッドをサポートしているモデルを表示
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            # print(f"  (詳細: {m.description})") # 詳細を見たい場合はコメントアウトを外す

    print("\n--- 🔢 Embeddingモデル (ベクトル化用) ---")
    # embedContent メソッドをサポートしているモデルを表示
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"- {m.name}")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")

print("\n========================================")
print("確認完了")