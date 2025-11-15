"""
Streamlit Web App để dự đoán Financial Sentiment Analysis
"""

import streamlit as st
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = "./models/finbert-trained/final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

def predict_sentiment(text: str):
    """Dự đoán sentiment cho một câu văn bản"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        label = id2label[pred_id]
        confidence = probs[pred_id].item()
    
    return label, confidence

def predict_batch(texts: list):
    """Dự đoán sentiment cho nhiều câu cùng lúc"""
    results = []
    for text in texts:
        label, confidence = predict_sentiment(text)
        results.append((label, confidence))
    return results

# Page config
st.set_page_config(
    page_title="Phân tích Cảm xúc Tài chính",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("FinBERT Financial Sentiment Classifier")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Giới thiệu")
    st.markdown("""
    Ứng dụng này sử dụng mô hình **FinBERT** đã được fine-tune để phân tích 
    cảm xúc tài chính trong văn bản.
    
    **Nhãn:**
    - 🟢 **Positive**: Triển vọng tài chính tích cực
    - 🔴 **Negative**: Triển vọng tài chính tiêu cực  
    - 🟡 **Neutral**: Câu trung tính hoặc thực tế
    
    **Mô hình:** FinBERT (yiyanghkust/finbert-tone)
    """)
    
    st.markdown("---")
    st.header("🔧 Cài đặt")
    show_confidence = st.checkbox("Hiển thị điểm tin cậy", value=True)
    show_details = st.checkbox("Hiển thị chi tiết dự đoán", value=False)

# Main content
tab1, tab2 = st.tabs(["📝 Phân tích đơn", "📄 Phân tích hàng loạt"])

with tab1:
    st.header("Phân tích một câu văn bản")
    
    # Text input
    text_input = st.text_area(
        "Nhập văn bản tài chính cần phân tích:",
        height=150,
        placeholder="Ví dụ: Công ty kỳ vọng tăng trưởng doanh thu mạnh trong năm tới."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Phân tích", type="primary", use_container_width=True)
    
    if predict_button and text_input:
        with st.spinner("Đang phân tích cảm xúc..."):
            try:
                label, confidence = predict_sentiment(text_input)
                
                # Display result
                st.markdown("### Kết quả")
                
                # Color coding based on label
                if label == "Positive":
                    css_class = "positive"
                    emoji = "🟢"
                    label_vn = "Tích cực"
                elif label == "Negative":
                    css_class = "negative"
                    emoji = "🔴"
                    label_vn = "Tiêu cực"
                else:
                    css_class = "neutral"
                    emoji = "🟡"
                    label_vn = "Trung tính"
                
                # Result box
                result_html = f"""
                <div class="prediction-box {css_class}">
                    <h3>{emoji} {label_vn}</h3>
                """
                if show_confidence:
                    result_html += f"<p><strong>Độ tin cậy:</strong> {confidence:.2%}</p>"
                result_html += "</div>"
                
                st.markdown(result_html, unsafe_allow_html=True)
                
                # Details
                if show_details:
                    with st.expander("📊 Chi tiết dự đoán"):
                        st.write(f"**Nhãn:** {label_vn} ({label})")
                        st.write(f"**Độ tin cậy:** {confidence:.4f}")
                        st.write(f"**Độ dài văn bản:** {len(text_input)} ký tự")
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    
    elif predict_button and not text_input:
        st.warning("⚠️ Vui lòng nhập văn bản cần phân tích.")

with tab2:
    st.header("Phân tích hàng loạt")
    
    # Batch input
    batch_input = st.text_area(
        "Nhập nhiều văn bản (mỗi dòng một câu):",
        height=200,
        placeholder="""Ví dụ:
Công ty kỳ vọng tăng trưởng doanh thu mạnh trong năm tới.
Doanh số giảm 10% trong quý thứ ba.
Thỏa thuận có hiệu lực đến năm 2008."""
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        batch_button = st.button("🔍 Phân tích hàng loạt", type="primary", use_container_width=True)
    
    if batch_button and batch_input:
        texts = [line.strip() for line in batch_input.split("\n") if line.strip()]
        
        if texts:
            with st.spinner(f"Đang phân tích {len(texts)} văn bản..."):
                try:
                    results = predict_batch(texts)
                    
                    st.markdown("### Kết quả")
                    
                    # Display results
                    for i, (text, (label, confidence)) in enumerate(zip(texts, results), 1):
                        # Color coding
                        if label == "Positive":
                            css_class = "positive"
                            emoji = "🟢"
                            label_vn = "Tích cực"
                        elif label == "Negative":
                            css_class = "negative"
                            emoji = "🔴"
                            label_vn = "Tiêu cực"
                        else:
                            css_class = "neutral"
                            emoji = "🟡"
                            label_vn = "Trung tính"
                        
                        # Result box
                        result_html = f"""
                        <div class="prediction-box {css_class}">
                            <h4>Văn bản {i}: {emoji} {label_vn}</h4>
                            <p><em>"{text}"</em></p>
                        """
                        if show_confidence:
                            result_html += f"<p><strong>Độ tin cậy:</strong> {confidence:.2%}</p>"
                        result_html += "</div>"
                        
                        st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Summary statistics
                    if show_details:
                        with st.expander("📊 Thống kê tổng hợp"):
                            labels = [label for _, (label, _) in results]
                            positive_count = labels.count("Positive")
                            negative_count = labels.count("Negative")
                            neutral_count = labels.count("Neutral")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🟢 Tích cực", positive_count)
                            with col2:
                                st.metric("🔴 Tiêu cực", negative_count)
                            with col3:
                                st.metric("🟡 Trung tính", neutral_count)
                            
                            # Average confidence
                            avg_confidence = sum(conf for _, (_, conf) in results) / len(results)
                            st.metric("Độ tin cậy trung bình", f"{avg_confidence:.2%}")
                
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
        else:
            st.warning("⚠️ Vui lòng nhập ít nhất một văn bản để phân tích.")
    
    elif batch_button and not batch_input:
        st.warning("⚠️ Vui lòng nhập văn bản cần phân tích.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Được hỗ trợ bởi FinBERT | Mô hình Phân tích Cảm xúc Tài chính</p>
    </div>
    """,
    unsafe_allow_html=True
)

