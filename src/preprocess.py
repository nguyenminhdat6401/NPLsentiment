import re
import pandas as pd
from underthesea import word_tokenize

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    # Chuyển chữ thường
    text = text.lower()
    # Xóa ký tự đặc biệt, giữ lại chữ cái và số
    text = re.sub(r'[^\w\s]', '', text)
    # Tách từ tiếng Việt (ví dụ: "sản phẩm" -> "sản_phẩm")
    text = word_tokenize(text, format="text")
    return text

def run_preprocessing(input_path, output_path):
    print(f"--- Đang đọc dữ liệu từ: {input_path} ---")
    df = pd.read_csv(input_path)
    
    # Xử lý các dòng trống
    df = df.dropna(subset=['comment', 'label'])
    
    print("--- Đang làm sạch văn bản... ---")
    df['comment_clean'] = df['comment'].apply(clean_text)
    
    # Lưu file sạch
    df.to_csv(output_path, index=False)
    print(f"--- Đã lưu dữ liệu sạch tại: {output_path} ---")

if __name__ == "__main__":
    # Chạy trực tiếp để xử lý file
    run_preprocessing('data/NPLpro.csv', 'data/processed_NPLpro.csv')