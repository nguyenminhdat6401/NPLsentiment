import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train():
    print("--- Đang tải dữ liệu sạch... ---")
    df = pd.read_csv('data/processed_NPLpro.csv')
    
    # Chuẩn bị dữ liệu
    X = df['comment_clean'].values.astype('U')
    y = df['label']
    
    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Biến đổi văn bản thành số (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Huấn luyện Logistic Regression
    print("--- Đang huấn luyện mô hình AI... ---")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Đánh giá kết quả
    y_pred = model.predict(X_test_tfidf)
    print("\nKẾT QUẢ HUẤN LUYỆN:")
    print(classification_report(y_test, y_pred))
    
    # Lưu mô hình và vectorizer vào file pkl
    joblib.dump({'model': model, 'vectorizer': vectorizer}, 'model/model.pkl')
    print("--- Đã lưu mô hình thành công vào folder model/ ---")

if __name__ == "__main__":
    train()