# app.py (最終安定版: @st.singleton導入)
import streamlit as st
import io
from PIL import Image
import google.generativeai as genai

# --- APIキーの設定 ---
try:
    # この部分はアプリ起動時に一度だけ実行される
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception:
    st.error("⚠️ エラー: Gemini APIキーが設定されていません。")
    st.info("Streamlit Community CloudのSecretsにキーを設定してください。")
    st.stop()

# --- @st.singleton: AIモデルを一度だけ初期化し、記憶し続ける ---
# これにより、スリープからの復帰時にモデルを再読み込みする必要がなくなり、クラッシュを防ぎます。
@st.singleton
def init_model():
    return genai.GenerativeModel('gemini-pro-vision')

# --- @st.cache_data: 同じ画像・プロンプトの解析結果をキャッシュする ---
@st.cache_data
def get_gemini_response(_model, _image_bytes, prompt):
    image = Image.open(io.BytesIO(_image_bytes))
    try:
        response = _model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        st.error(f"AIとの通信中にエラーが発生しました: {e}")
        return None

# --- プロンプト定義 ---
PROMPT_BASE = """この画像に含まれる手書きの文字を、可能な限り正確に全て書き起こしてください。書き起こし以外の、画像に関する説明やコメント、補足情報は一切含めないでください。"""
PROMPT_VARIANT = """画像内の手書き文字を完全に書き起こしてください。誤字脱字、判読困難な文字があった場合でも、その文字の意図を汲み取り、もっともらしいテキストに変換してください。出力には、書き起こされたテキストのみを含めてください。"""
FINAL_JUDGEMENT_PROMPT = """あなたは優秀な編集者です。提示された複数のOCR結果と、元の画像を注意深く見比べ、全ての情報を統合し、最も正確で、元の手書きの意図を完璧に反映した**最終的な書き起こしテキストを一つだけ生成してください。**余計な説明、前置き、コメントは一切不要です。
---
【OCR結果1】
{text1}
---
【OCR結果2】
{text2}
---
上記の指示に従い、元の画像から最終的な書き起こしテキストを生成してください。"""

# --- Streamlit UIの構築 ---
st.set_page_config(page_title="Moji Scan", layout="centered")
st.title("📝 Moji Scan")
st.markdown("手書き文字の画像をアップロードすると、AIがテキストに書き起こします。")

# Singletonで初期化されたモデルを取得
model = init_model()

uploaded_file = st.file_uploader(
    "画像ファイルを選択してください",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="アップロードされた画像", use_column_width=True)

    with st.spinner("AIが手書き文字を解析中です..."):
        st.info("ステップ1/2: 異なる方法で文字を解析しています...")
        response1 = get_gemini_response(model, image_bytes, PROMPT_BASE)
        response2 = get_gemini_response(model, image_bytes, PROMPT_VARIANT)

        if response1 is None or response2 is None:
            st.error("解析を中断しました。時間をおいて再度お試しください。")
        elif response1 == response2:
            final_result = response1
            st.success("解析が完了しました。（結果が一致したため高精度です）")
            st.text_area("以下のテキストをコピーしてご利用ください:", final_result, height=250)
        else:
            st.info("ステップ2/2: 結果の精度を高めるため、追加の検証を行っています...")
            final_prompt = FINAL_JUDGEMENT_PROMPT.format(text1=response1, text2=response2)
            final_result = get_gemini_response(model, image_bytes, final_prompt)

            if final_result is None:
                st.error("最終検証に失敗しました。最初の解析結果を表示します。")
                final_result = response1
            else:
                st.success("検証が完了し、最終的な結果を生成しました。")
            st.text_area("以下のテキストをコピーしてご利用ください:", final_result, height=250)

st.markdown("---")
st.markdown("<div style='text-align: center;'>Powered by Google Gemini API</div>", unsafe_allow_html=True)