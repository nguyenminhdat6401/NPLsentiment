import joblib
import sys
import os

# Thêm đường dẫn để có thể gọi hàm clean_text từ file preprocess.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

def get_prediction(text):
    # Load "trí tuệ" đã lưu
    if not os.path.exists('model/model.pkl'):
        return "Lỗi: Bạn cần chạy train.py trước để tạo file model.pkl"
    
    saved_data = joblib.load('model/model.pkl')
    model = saved_data['model']
    vectorizer = saved_data['vectorizer']
    
    # Sơ chế câu nhập vào
    cleaned = clean_text(text)
    # Chuyển thành số
    vectorized = vectorizer.transform([cleaned])
    # Dự đoán
    result = model.predict(vectorized)
    
    return result[0]

if __name__ == "__main__":
    print("--- HỆ THỐNG PHÂN TÍCH CẢM XÚC ---")
    while True:
        user_input = input("Nhập bình luận của khách (hoặc 'q' để thoát): ")
        if user_input.lower() == 'q':
            break
        label = get_prediction(user_input)
        print(f"==> Kết quả: {label}\n")