# app.py (二重チェック機能 復元版)
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
    st.info("Streamlit Community CloudのSecretsにキーを設定してください。")
    st.stop()

# --- Geminiモデルと3種類のプロンプトを定義 ---
model = genai.GenerativeModel('gemini-2.0-flash')

PROMPT_BASE = """
この画像に含まれる手書きの文字を、可能な限り正確に全て書き起こしてください。
文字のかすれ、にじみ、傾き、または判読しにくい部分があっても、文脈から最も適切と思われる内容を推測し、テキストとして出力してください。
句読点や改行も元の手書き文字に忠実に再現してください。
書き起こし以外の、画像に関する説明やコメント、補足情報は一切含めないでください。
結果は純粋なテキスト形式で提供してください。
"""
PROMPT_VARIANT = """
画像内の手書き文字を完全に書き起こしてください。
誤字脱字、判読困難な文字があった場合でも、その文字の意図を汲み取り、もっともらしいテキストに変換してください。
出力には、書き起こされたテキストのみを含め、それ以外の説明やコメントは一切含めないでください。
"""
FINAL_JUDGEMENT_PROMPT = """
あなたは非常に優秀な編集者です。これから、一つの手書き画像に対して行った、複数のOCR（文字認識）結果を提示します。
これらのOCR結果には、それぞれわずかな誤りや解釈の違いが含まれている可能性があります。
提示された複数のOCR結果と、元の画像を注意深く見比べてください。
そして、全ての情報を統合し、最も正確で、元の手書きの意図を完璧に反映した**最終的な書き起こしテキストを一つだけ生成してください。**
- 元の画像の改行、箇条書きのスタイルを忠実に再現してください。
- 余計な説明、前置き、コメントは一切不要です。
- 最終的な書き起こしテキストのみを出力してください。
---
【OCR結果1】
{text1}
---
【OCR結果2】
{text2}
---
上記の指示に従い、元の画像から最終的な書き起こしテキストを生成してください。
"""

# --- 関数定義 ---
def get_gemini_response(image, prompt):
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"エラーが発生しました: {e}"

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

    final_result = ""
    with st.spinner("AIが手書き文字を解析中です..."):
        # ステップ1: 異なるプロンプトで2つの書き起こし候補を生成
        st.info("ステップ1/2: 異なる方法で文字を解析しています...")
        response1 = get_gemini_response(image, PROMPT_BASE)
        response2 = get_gemini_response(image, PROMPT_VARIANT)

        # エラーチェック
        if "エラー" in response1 or "エラー" in response2:
            st.error("解析中にエラーが発生しました。しばらくしてから再度お試しください。")
            st.stop()

        # ステップ2: 2つの結果を比較
        if response1 == response2:
            final_result = response1
            st.success("解析が完了しました。（結果が一致したため高精度です）")
        else:
            # ステップ3: 結果が異なる場合、AIに最終判断を依頼
            st.info("ステップ2/2: 結果の精度を高めるため、追加の検証を行っています...")
            final_prompt = FINAL_JUDGEMENT_PROMPT.format(text1=response1, text2=response2)
            final_result = get_gemini_response(image, final_prompt)
            st.success("検証が完了し、最終的な結果を生成しました。")

    st.subheader("書き起こし結果")
    st.text_area("以下のテキストをコピーしてご利用ください:", final_result, height=250)

st.markdown("---")
st.markdown("<div style='text-align: center;'>Powered by Google Gemini API</div>", unsafe_allow_html=True)