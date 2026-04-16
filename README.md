# 🏛️ AI HÀNH CHÍNH CÔNG
## Hệ thống tư vấn thủ tục hành chính công thông minh

---

## Mục đích sử dụng
Chạy train model AI, ML, LLM cho hệ thống tư vấn thủ tục hành chính công

---

## Thông tin về cấu hình, thư viện
- **CUDA Version**: 12.2
- **NVIDIA (R) Cuda compiler driver**: release 11.5, V11.5.119
- **GPU**: 80GB, **CPU**: 64GB
- **Conda**: 23.1.0
- **Pinecone**: Cloud vector database for embeddings ( cần PINECONE_API_KEY)

---

## Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## Chạy train
```bash
python3 main.py
```

---

## Các tham số và vị trí folder data

### Folder data
- `./data/` - Thư mục chứa dữ liệu (tự động tạo)
- `./models/` - Thư mục chứa models (tự động tạo)
- `./audio_output/` - Thư mục chứa file âm thanh đầu ra (tự động tạo)

### Tham số cấu hình
- **Model LLM**: Qwen/Qwen2.5-3B-Instruct
- **4-bit Quantization**: Giảm bộ nhớ mà vẫn giữ độ chính xác
- **Embeddings Model**: keepitreal/vietnamese-sbert (768 dimensions)
- **Vector Store**: Pinecone (Dense Search) + BM25 (Sparse Search)
  - Index name: ai-hanh-chinh-rag
  - Metric: cosine
  - Serverless on AWS us-east-1
  - Cần thiết lập `PINECONE_API_KEY` environment variable
- **Voice AI**: Whisper (base) + Edge-TTS
- **Temperature**: 0.1 - 0.7 (tự điều chỉnh theo độ phức tạp câu hỏi)
- **Max Tokens**: 1024 cho câu trả lời

---

## Mô tả các chức năng chính

### 1. Hybrid RAG System
- **Dense Search**: Pinecone + Vietnamese SBERT embeddings
- **Sparse Search**: BM25 keyword matching
- **Query Expansion**: Mở rộng câu hỏi với từ đồng nghĩa
- **Re-ranking**: Sắp xếp kết quả theo độ liên quan
- **Pinecone Integration**: Embeddings được lưu trữ trên Pinecone cloud database

### 2. Phân loại tình huống (50+ situations)
- Ensemble Classifier (Random Forest + Logistic Regression)
- Phân loại chính xác tình huống người dùng
- Áp dụng TF-IDF vectorization

### 3. Voice AI
- **Speech-to-Text**: OpenAI Whisper model base cho tiếng Việt
- **Text-to-Speech**: Microsoft Edge-TTS với giọng HoaiMy
- Hỗ trợ người lớn tuổi không biết gõ phím

### 4. 3 AI Personas
- 👩‍💼 **Chị Thuong** - Thân thiện, gần gũi, dùng từ ngữ bình dân
- 👨‍💼 **Anh Chuyen** - Chuyên nghiệp, ngắn gọn, súc tích
- 👩‍🏫 **Cô Chi Tiet** - Chi tiết, từng bước, cẩn thận

### 5. Context Memory
- Nhớ lịch sử chat (tối đa 10 lượt)
- Hiểu ngữ cảnh từ các câu hỏi trước
- Câu trả lời phù hợp với ngữ cảnh

### 6. Chain of Thought
- Multi-step reasoning prompts
- Phân tích từng bước trước khi trả lời

---

## Danh sách thủ tục hành chính (37+ procedures)

### Nhân sự (6)
- KHAI_SINH - Đăng ký khai sinh
- KHAI_SINH_QUA_HAN - Khai sinh quá hạn
- KHAI_TU - Đăng ký khai tử
- KET_HON - Đăng ký kết hôn
- LY_HON - Ly hôn thuận tình
- KET_HON_NGOAI - Kết hôn người nước ngoài

### CCCD - Hộ khẩu (8)
- CCCD_CAP - Cấp CCCD lần đầu
- CCCD_DOI - Đổi CCCD
- CCCD_CAP_LAI - Cấp lại CCCD khi mất
- HO_KHAU_TACH - Tách sổ hộ khẩu
- HO_KHAU_NHAP - Nhập sổ hộ khẩu
- TAM_TRU - Đăng ký tạm trú
- TRU_SO - Đăng ký trú quán
- TAM_VANG - Đăng ký tạm vắng

### Bằng lái (7)
- BANG_LAI_A1 - Cấp đổi bằng lái A1
- BANG_LAI_A1_CAP_MOI - Cấp mới bằng lái A1
- BANG_LAI_A2 - Cấp bằng lái A2
- BANG_LAI_B1 - Cấp bằng lái B1
- BANG_LAI_B2 - Cấp bằng lái B2
- BANG_LAI_DOI - Đổi bằng lái
- BANG_LAI_CAP_LAI - Cấp lại bằng lái

### Và nhiều thủ tục khác...

---

## Cấu trúc source code
```
AIHanhChinh/
├── main.py              # File chính để chạy
├── requirements.txt     # Danh sách thư viện
├── README.md           # Hướng dẫn sử dụng
├── data/               # Thư mục data (tự động tạo)
├── models/             # Thư mục models (tự động tạo)
└── audio_output/       # Thư mục audio (tự động tạo)
```

---

## Lưu ý quan trọng
- Code được viết để chạy không cần tương tác trực tiếp
- Có thể thực thi từ đầu đến cuối mà không cần sự can thiệp
- Tự động tải datasets từ HuggingFace Hub
- Tự động tạo các thư mục cần thiết

### Thiết lập Pinecone API Key
Trước khi chạy, cần thiết lập `PINECONE_API_KEY`:
```bash
# Linux/Mac
export PINECONE_API_KEY='your-api-key-here'

# Windows CMD
set PINECONE_API_KEY=your-api-key-here

# Windows PowerShell
$env:PINECONE_API_KEY='your-api-key-here'
```

Hoặc nhập trực tiếp vào code (không khuyến khích cho production).

Lấy API key tại: https://app.pinecone.io/keys

---

## Ví dụ câu hỏi
- ✅ "Làm bằng lái xe máy cần giấy tờ gì?"
- ✅ "Khai sinh quá hạn phải làm sao?"
- ✅ "Đổi CCCD ở đâu và bao nhiêu tiền?"
- ✅ "Tách hộ khẩu mất bao lâu?"
- ✅ "Đăng ký xe ô tô mới cần những gì?"

---

## Ghi chú
- Ứng dụng sử dụng Qwen 2.5 3B - Model open-source hiểu tiếng Việt tốt
- Voice AI sử dụng Whisper (OpenAI) và Edge-TTS (Microsoft)
- Hệ thống RAG kết hợp Dense và Sparse search để tối ưu độ chính xác
- Context Memory giúp AI nhớ ngữ cảnh hội thoại
- Dynamic Temperature tự điều chỉnh độ sáng tạo dựa trên độ phức tạp câu hỏi
