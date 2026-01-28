# app.py 
import streamlit as st
import io
from PIL import Image
import google.generativeai as genai
import json
from difflib import SequenceMatcher

# --- 編集距離を計算する関数 ---
def calculate_similarity(text1, text2):
    """2つのテキスト間の類似度を計算（0-100%）"""
    # 改行を除外
    text1_clean = text1.replace('\n', '').replace('\r', '')
    text2_clean = text2.replace('\n', '').replace('\r', '')
    matcher = SequenceMatcher(None, text1_clean, text2_clean)
    similarity = matcher.ratio() * 100
    return similarity

def calculate_edit_distance(text1, text2):
    """レーベンシュタイン距離を計算"""
    # 改行を除外
    text1 = text1.replace('\n', '').replace('\r', '')
    text2 = text2.replace('\n', '').replace('\r', '')
    len1, len2 = len(text1), len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len1][len2]

# --- APIキーの設定 ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception:
    st.error("エラー: Gemini APIキーが設定されていません。")
    st.info("Streamlit Community CloudのSecretsにキーを設定してください。")
    st.stop()

# --- @st.cache_resource: AIを一度だけ準備し、リソースとして記憶 ---
@st.cache_resource
def init_model():
    return genai.GenerativeModel('gemma-3-27b-it')

# --- @st.cache_data: 結果をデータとしてキャッシュ ---
@st.cache_data
def get_gemini_response(image_bytes, prompt):
    model = init_model() # ここでキャッシュされたモデルを呼び出す
    image = Image.open(io.BytesIO(image_bytes))
    try:
        response = model.generate_content([prompt, image])
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
st.title("Moji Scan")
st.markdown("手書き文字の画像をアップロードすると、AIがテキストに書き起こします。")

uploaded_file = st.file_uploader(
    "画像ファイルを選択してください",
    type=["png", "jpg", "jpeg"]
)

# 修正点
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="アップロードされた画像", use_column_width=True)

    with st.spinner("AIが手書き文字を解析中です..."):
        st.info("ステップ1/3: Gemma単体での書き起こしを実行中...")
        response_single = get_gemini_response(image_bytes, PROMPT_BASE)
        
        st.info("ステップ2/3: 異なる方法で文字を解析しています...")
        response1 = get_gemini_response(image_bytes, PROMPT_BASE)
        response2 = get_gemini_response(image_bytes, PROMPT_VARIANT)

        if response1 is None or response2 is None or response_single is None:
            st.error("解析を中断しました。時間をおいて再度お試しください。")
        elif response1 == response2:
            final_result = response1
            st.success("解析が完了しました。（結果が一致したため高精度です）")
            
            # 複合方式との比較表示
            st.markdown("### 書き起こし結果の比較")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gemma単体")
                st.text_area("", response_single, height=200, label_visibility="collapsed", key="baseline_area")
            with col2:
                st.subheader("最終結果（複合方式）")
                st.text_area("", final_result, height=200, label_visibility="collapsed", key="final_area")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_area("コピー用：", final_result, height=100, label_visibility="collapsed")
            with col2:
                if st.button("copy", key="copy_button_1", use_container_width=True):
                    st.write(final_result)
                    st.success("コピーしました！")
            
            # 正しいテキスト入力と精度評価
            st.markdown("---")
            st.markdown("### 精度の評価")
            correct_text = st.text_area("正しいテキストを入力してください：", height=100, key="correct_text_1")
            
            if correct_text:
                st.markdown("#### 編集距離による精度評価")
                edit_dist_single = calculate_edit_distance(response_single, correct_text)
                edit_dist_final = calculate_edit_distance(final_result, correct_text)
                similarity_single = calculate_similarity(response_single, correct_text)
                similarity_final = calculate_similarity(final_result, correct_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Gemma単体", f"編集距離: {edit_dist_single}", f"類似度: {similarity_single:.1f}%")
                with col2:
                    st.metric("複合方式", f"編集距離: {edit_dist_final}", f"類似度: {similarity_final:.1f}%")
        else:
            st.info("ステップ3/3: 結果の精度を高めるため、追加の検証を行っています...")
            final_prompt = FINAL_JUDGEMENT_PROMPT.format(text1=response1, text2=response2)
            final_result = get_gemini_response(image_bytes, final_prompt)

            if final_result is None:
                st.error("最終検証に失敗しました。最初の解析結果を表示します。")
                final_result = response1
            else:
                st.success("検証が完了し、最終的な結果を生成しました。")
            
            # 複合方式との比較表示
            st.markdown("### 書き起こし結果の比較")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gemma単体")
                st.text_area("", response_single, height=200, label_visibility="collapsed", key="baseline_area_2")
            with col2:
                st.subheader("最終結果（複合方式）")
                st.text_area("", final_result, height=200, label_visibility="collapsed", key="final_area_2")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_area("コピー用：", final_result, height=100, label_visibility="collapsed")
            with col2:
                if st.button("📋 コピー", key="copy_button_2", use_container_width=True):
                    st.write(final_result)
                    st.success("コピーしました！")
            
            # 正しいテキスト入力と精度評価
            st.markdown("---")
            st.markdown("### 精度の評価")
            correct_text = st.text_area("正しいテキストを入力してください：", height=100, key="correct_text_2")
            
            if correct_text:
                st.markdown("#### 編集距離による精度評価")
                edit_dist_single = calculate_edit_distance(response_single, correct_text)
                edit_dist_final = calculate_edit_distance(final_result, correct_text)
                similarity_single = calculate_similarity(response_single, correct_text)
                similarity_final = calculate_similarity(final_result, correct_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Gemma単体", f"編集距離: {edit_dist_single}", f"類似度: {similarity_single:.1f}%")
                with col2:
                    st.metric("複合方式", f"編集距離: {edit_dist_final}", f"類似度: {similarity_final:.1f}%")
        
        st.success("全ての解析が完了しました。")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Powered by Google Gemini API</div>", unsafe_allow_html=True)
