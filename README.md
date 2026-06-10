# AIDEOM-VN Final Project

Dashboard và mã nguồn cho bộ bài tập cuối kỳ **Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI**.

## Cấu trúc thư mục theo đề xuất của giảng viên

```text
aideom_vn/
├── venv/                    # Môi trường ảo Python, tự tạo local, KHÔNG upload GitHub
├── app.py                   # Streamlit dashboard chính
├── data/                    # Dữ liệu CSV gốc
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
├── notebooks/               # Jupyter notebooks cho từng bài
│   ├── bai01_cobb_douglas.ipynb
│   ├── bai02_lp_phan_bo.ipynb
│   └── ... bai12_tich_hop_he_thong.ipynb
├── src/                     # Mã Python tái sử dụng
│   ├── __init__.py
│   ├── data_loader.py
│   ├── optimization.py
│   ├── rl_env.py
│   ├── charts.py
│   └── reporting.py
├── outputs/                 # Kết quả: bảng, biểu đồ, mô hình lưu
├── reports/                 # Báo cáo Word/PDF nộp bài
├── tests/                   # Unit tests cơ bản
├── requirements.txt
├── requirements_full.txt
└── README.md
```

> Lưu ý: thư mục `venv/` chỉ tạo trên máy cá nhân, không đưa vào file nộp hoặc GitHub.

## Cài đặt và chạy local

```powershell
cd "C:\Users\ADMIN\Downloads\aideom_vn"
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Nếu không dùng môi trường ảo:

```powershell
python -m pip install --user -r requirements.txt
python -m streamlit run app.py
```

## Chạy kiểm thử

```powershell
python -m pytest tests
```

## Deploy Streamlit Cloud

- Repository: repo GitHub của bạn
- Branch: `main`
- Main file path: `app.py`

Không upload các thư mục tự sinh như `__pycache__`, `.pytest_cache`, `venv`.

## Nội dung mô hình

Dashboard gồm 14 tab:
1. Tổng quan dữ liệu
2. Bài 1 – Cobb-Douglas, TFP, phân rã tăng trưởng, dự báo 2030
3. Bài 2 – LP phân bổ ngân sách
4. Bài 3 – Priority ngành và độ nhạy
5. Bài 4 – LP vùng × ngân sách có công bằng
6. Bài 5 – MIP chọn dự án
7. Bài 6 – TOPSIS vùng
8. Bài 7 – Pareto
9. Bài 8 – Tối ưu động 2026–2035
10. Bài 9 – AI và lao động NetJob
11. Bài 10 – Stochastic programming, VSS, EVPI
12. Bài 11 – Q-learning 81 trạng thái, 5 hành động
13. Bài 12 – So sánh kịch bản S1–S5
14. Phân tích chính sách theo ngành
