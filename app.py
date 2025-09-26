# app.py
import streamlit as st
import io
from PIL import Image
import google.generativeai as genai

# --- APIキーの設定 ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception:
    st.error("⚠️ エラー: Gemini APIキーが設定されていません。")
    st.info("この後、Streamlit Community CloudのSecretsにキーを設定してください。")
    st.stop()

# --- Geminiモデルとプロンプトの定義 ---
model = genai.GenerativeModel('gemini-2.0-flash')
PROMPT = "この画像に含まれる手書きの文字を、可能な限り正確に全て書き起こしてください。書き起こし以外の、画像に関する説明やコメント、補足情報は一切含めないでください。"

# --- Streamlit UIの構築 ---
st.set_page_config(page_title="Moji Scan", layout="centered")
st.title("📝 Moji Scan")
st.markdown("手書き文字の画像をアップロードすると、AIがテキストに書き起こします。")

uploaded_file = st.file_uploader(
    "画像ファイルを選択してください",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption="アップロードされた画像", use_column_width=True)

    with st.spinner("AIが手書き文字を解析中です..."):
        try:
            response = model.generate_content([PROMPT, image])
            result_text = response.text.strip()
            st.subheader("書き起こし結果")
            st.text_area("以下のテキストをコピーしてご利用ください:", result_text, height=250)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Powered by Google Gemini API</div>", unsafe_allow_html=True)