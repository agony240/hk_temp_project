# 香港氣溫預測專案 (hk_temp_project)

## 專案簡介
本專案利用 **RNN** 與 **LSTM** 模型，對香港未來 7 日的氣溫進行預測。  
透過 Flask 建立網頁介面，使用者可以在瀏覽器中查看預測結果，並以表格及圖表方式呈現。

---

## 功能特色
- 📊 顯示未來 7 日的氣溫預測（攝氏 °C）
- 🔄 支援 RNN 與 LSTM 模型比較
- 🌐 Flask 網頁介面，簡單易用
- 🎨 使用 Bootstrap 美化介面，表格與按鈕清晰易讀

---

## 專案結構

hk_temp_project/
│ 
├── app.py # Flask 主程式 
├── templates/ # HTML 模板 (Bootstrap) 
├── static/ # 圖片、CSS、JS 
├── models/ # 儲存訓練好的模型檔案 
├── data/ # 原始數據 
├── train_rnn.py # RNN 訓練程式 
├── train_model.py # LSTM 訓練程式 
├── evaluate.py # 模型評估程式 
└── requirements.txt # 套件需求


---

## 安裝與執行
### 1. 建立虛擬環境
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows

## 安裝套件
pip install -r requirements.txt

## 啟動 Flask
python app.py
