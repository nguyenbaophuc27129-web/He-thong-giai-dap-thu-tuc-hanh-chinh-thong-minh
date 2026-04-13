# ============================================================
# 🏛️ AI HÀNH CHÍNH CÔNG COMPLETE - MAIN.PY
# 100% CODE TỪ JUPYTER NOTEBOOK - GIỮ NGUYÊN TOÀN BỘ
# File chạy độc lập, không cần tương tác, có thể thực thi từ đầu đến cuối
# ============================================================

# ============================================================
# PHẦN 1: IMPORT & CẤU HÌNH
# ============================================================

import os
import sys
import warnings
import json
import time
import random
import gc
import re
import asyncio
from functools import lru_cache
from typing import List, Dict, Tuple, Any
import hashlib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Cài đặt warnings
warnings.filterwarnings('ignore')

# Tạo thư mục
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./audio_output', exist_ok=True)

# ============================================================
# PHẦN 2: LOAD DATASETS
# ============================================================

print('='*70)
print('📦 Đang tải datasets...')
print('='*70)

# Dataset 1: Hirine - 5733 samples
try:
    from datasets import load_dataset
    ds1 = load_dataset("hirine/dataset-thu-tuc-hanh-chinh-5733-samples")
    df1 = pd.DataFrame(ds1['train'])
    print(f'   ✅ Dataset 1: {len(df1):,} samples (Hirine)')
except Exception as e:
    df1 = pd.DataFrame(columns=['title', 'text'])
    print(f'   ⚠️ Dataset 1: Skip ({str(e)})')

# Dataset 2: Large Legal Queries
try:
    ds2 = load_dataset("phamson02/large-vi-legal-queries", split="train")
    df2 = ds2.to_pandas()[['title', 'context']].rename(columns={'context': 'text'})
    print(f'   ✅ Dataset 2: {len(df2):,} samples (Legal Queries)')
except Exception as e:
    df2 = pd.DataFrame(columns=['title', 'text'])
    print(f'   ⚠️ Dataset 2: Skip ({str(e)})')

print(f'   📊 Total: {len(df1) + len(df2):,} samples')
print()

# ============================================================
# PHẦN 3: LOAD EMBEDDINGS & FAISS
# ============================================================

print('='*70)
print('🔤 Đang tải Embeddings & Vector Store...')
print('='*70)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='keepitreal/vietnamese-sbert',
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print('   ✅ Embeddings loaded')

# Build documents
all_docs = []

# Từ Dataset 1
for _, row in df1.iterrows():
    doc = Document(
        page_content=f"{row['title']}\n{row['text']}",
        metadata={'source': 'ds1', 'title': row['title']}
    )
    all_docs.append(doc)

# Từ Dataset 2
for _, row in df2.iterrows():
    doc = Document(
        page_content=f"{row['title']}\n{row['text']}",
        metadata={'source': 'ds2', 'title': row['title']}
    )
    all_docs.append(doc)

# Tạm thời chưa có DICHVUCONG_DATA, sẽ thêm sau

# Create FAISS vector store
vector_db = FAISS.from_documents(all_docs, embeddings)
print(f'   ✅ FAISS created: {len(all_docs)} documents')
print()

# ============================================================
# PHẦN 4: DỮ LIỆU THỦ TỤC HÀNH CHÍNH (DICHVUCONG_DATA)
# ============================================================

DICHVUCONG_DATA = {
    # NHÂN SỤ (6)
    "KHAI_SINH": {"code":"KHAI_SINH","name":"Đăng ký khai sinh","docs":"Giấy khai sinh, CCCD cha mẹ, Sổ hộ khẩu","coquan":"UBND xã/phường","time":"1-3 ngày","lephi":"Miễn phí","note":"Làm trong 60 ngày kể từ khi sinh","steps":"1. Chuẩn bị: Giấy khai sinh, CCCD cha mẹ, Sổ hộ khẩu. 2. Đến UBND xã/phường nơi cư trú. 3. Nộp hồ sơ. 4. Nhận kết quả sau 1-3 ngày.","category":"NHAN_SU"},
    "KHAI_SINH_QUA_HAN": {"code":"KHAI_SINH_QUA_HAN","name":"Khai sinh quá hạn","docs":"Giấy khai sinh, CCCD, Sổ hộ khẩu, Đơn giải trình","coquan":"UBND xã/phường","time":"3-7 ngày","lephi":"Miễn phí","note":"Quá 60 ngày cần đơn giải trình","steps":"1. Viết đơn giải trình. 2. Xin xác nhận tổ dân phố. 3. Đến UBND nộp hồ sơ.","category":"NHAN_SU"},
    "KHAI_TU": {"code":"KHAI_TU","name":"Đăng ký khai tử","docs":"Giấy báo tử, CCCD người chết, Sổ hộ khẩu","coquan":"UBND xã/phường","time":"1-3 ngày","lephi":"Miễn phí","note":"Làm trong 60 ngày kể từ khi chết","steps":"1. Lấy giấy báo tử. 2. Chuẩn bị hồ sơ. 3. Đến UBND nơi người chết cư trú.","category":"NHAN_SU"},
    "KET_HON": {"code":"KET_HON","name":"Đăng ký kết hôn","docs":"CCCD cả 2 bên, Sổ hộ khẩu, Giấy xác nhận tình trạng hôn nhân","coquan":"UBND xã/phường","time":"1-3 ngày","lephi":"Miễn phí","note":"Trong 30 ngày có giấy xác nhận","steps":"1. Xin giấy xác nhận tình trạng hôn nhân. 2. Cả 2 đến UBND. 3. Đăng ký kết hôn.","category":"NHAN_SU"},
    "LY_HON": {"code":"LY_HON","name":"Ly hôn thuận tình","docs":"Đơn xin ly hôn, Giấy kết hôn, CCCD, Sổ hộ khẩu","coquan":"Tòa án","time":"2-4 tháng","lephi":"300.000đ","note":"Cả 2 bên đồng ý","steps":"1. Soạn đơn xin ly hôn. 2. Nộp tại Tòa án. 3. Tòa án thụ lý và ra quyết định.","category":"NHAN_SU"},
    "KET_HON_NGOAI": {"code":"KET_HON_NGOAI","name":"Kết hôn người nước ngoài","docs":"CCCD, Hộ chiếu, Giấy xác nhận, Giấy hợp pháp hóa lãnh sự","coquan":"UBND quận/huyện","time":"5-7 ngày","lephi":"Miễn phí","note":"Cần hợp pháp hóa lãnh sự","steps":"1. Hợp pháp hóa giấy tờ. 2. Xin xác nhận. 3. Đăng ký tại UBND quận.","category":"NHAN_SU"},

    # CCCD - HỘ KHẨU (8)
    "CCCD_CAP": {"code":"CCCD_CAP","name":"Cấp CCCD lần đầu","docs":"Giấy khai sinh, Sổ hộ khẩu","coquan":"Công an quận/huyện","time":"7-15 ngày","lephi":"Miễn phí","note":"Từ 14 tuổi","steps":"1. Chuẩn bị giấy khai sinh, sổ hộ khẩu. 2. Đến Công an quận/huyện. 3. Làm thủ tục cấp CCCD.","category":"CCCD_HK"},
    "CCCD_DOI": {"code":"CCCD_DOI","name":"Đổi CCCD","docs":"CCCD cũ, Giấy khai sinh","coquan":"Công an quận/huyện","time":"7-15 ngày","lephi":"Miễn phí","note":"Đổi khi hư hỏng/sai thông tin","steps":"1. Mang CCCD cũ. 2. Đến Công an nơi thường trú. 3. Làm thủ tục đổi.","category":"CCCD_HK"},
    "CCCD_CAP_LAI": {"code":"CCCD_CAP_LAI","name":"Cấp lại CCCD khi mất","docs":"Giấy khai sinh, Sổ hộ khẩu, Đơn xin cấp lại","coquan":"Công an quận/huyện","time":"7-15 ngày","lephi":"Miễn phí","note":"Báo mất và xin cấp lại","steps":"1. Làm đơn báo mất. 2. Đến Công an. 3. Xin cấp lại CCCD.","category":"CCCD_HK"},
    "HO_KHAU_TACH": {"code":"HO_KHAU_TACH","name":"Tách sổ hộ khẩu","docs":"Sổ hộ khẩu, CCCD, Đơn xin tách, Giấy tờ nơi ở mới","coquan":"Công an quận/huyện","time":"3-7 ngày","lephi":"Miễn phí","note":"Cần đồng ý chủ hộ","steps":"1. Viết đơn xin tách hộ khẩu. 2. Xin chữ ký chủ hộ. 3. Đến Công án làm thủ tục.","category":"CCCD_HK"},
    "HO_KHAU_NHAP": {"code":"HO_KHAU_NHAP","name":"Nhập sổ hộ khẩu","docs":"Sổ hộ khẩu, CCCD, Đơn xin nhập, Giấy tờ nơi ở","coquan":"Công an quận/huyện","time":"3-7 ngày","lephi":"Miễn phí","note":"Cần đồng ý chủ hộ mới","steps":"1. Viết đơn xin nhập. 2. Xin đồng ý chủ hộ mới. 3. Đến Công an làm thủ tục.","category":"CCCD_HK"},
    "TAM_TRU": {"code":"TAM_TRU","name":"Đăng ký tạm trú","docs":"CCCD, Đơn xin tạm trú, Giấy tờ nơi ở","coquan":"Công an xã/phường","time":"3-5 ngày","lephi":"Miễn phí","note":"Tối đa 2 năm","steps":"1. Lấy giấy xác nhận của chủ nhà. 2. Viết đơn xin tạm trú. 3. Đến Công an xã/phường.","category":"CCCD_HK"},
    "TRU_SO": {"code":"TRU_SO","name":"Đăng ký trú quán","docs":"CCCD, Đơn xin trú quán, Xác nhận tổ dân phố","coquan":"Công an xã/phường","time":"3-5 ngày","lephi":"Miễn phí","note":"Cho người không có hộ khẩu","steps":"1. Xin xác nhận tổ dân phố. 2. Viết đơn xin trú quán. 3. Đến Công an.","category":"CCCD_HK"},
    "TAM_VANG": {"code":"TAM_VANG","name":"Đăng ký tạm vắng","docs":"CCCD, Sổ hộ khẩu, Đơn xin","coquan":"Công an","time":"1-3 ngày","lephi":"Miễn phí","note":"Khi đi khỏi nơi thường trú trên 30 ngày","steps":"1. Làm đơn xin báo tạm vắng. 2. Nộp tại Công an nơi thường trú.","category":"CCCD_HK"},

    # BẰNG LÁI (7)
    "BANG_LAI_A1": {"code":"BANG_LAI_A1","name":"Cấp đổi bằng lái A1","docs":"CCCD, Giấy khai sinh (nếu dưới 18), 1 ảnh 3x4","coquan":"Sở GTVT","time":"7-10 ngày","lephi":"Miễn phí","note":"Từ 18 tuổi, đổi bằng không cần thi lại","steps":"1. Chuẩn bị CCCD, ảnh. 2. Đến Sở GTVT hoặc trung tâm sát hạch. 3. Nộp hồ sơ và nhận bằng mới.","category":"GIAO_THONG"},
    "BANG_LAI_A1_CAP_MOI": {"code":"BANG_LAI_A1_CAP_MOI","name":"Cấp mới bằng lái A1","docs":"CCCD, Giấy khai sinh, Học lý, Giấy khám sức khỏe","coquan":"Sở GTVT","time":"15-20 ngày","lephi":"110.000đ","note":"Phải thi sát hạch","steps":"1. Học lý thuyết và sa hình. 2. Đăng ký thi. 3. Thi sát hạch. 4. Nhận bằng khi đỗ.","category":"GIAO_THONG"},
    "BANG_LAI_A2": {"code":"BANG_LAI_A2","name":"Cấp bằng lái A2","docs":"CCCD, Học lý, Giấy khám sức khỏe","coquan":"Sở GTVT","time":"15-20 ngày","lephi":"150.000đ","note":"Xe mô tô trên 175cc","steps":"1. Học lý thuyết và sa hình A2. 2. Đăng ký thi. 3. Thi và nhận bằng.","category":"GIAO_THONG"},
    "BANG_LAI_B1": {"code":"BANG_LAI_B1","name":"Cấp bằng lái B1","docs":"CCCD, Học lý, Giấy khám sức khỏe","coquan":"Sở GTVT","time":"20-25 ngày","lephi":"350.000đ","note":"Ô tô dưới 9 chỗ","steps":"1. Học lý thuyết, sa hình, đường trường. 2. Đăng ký thi B1. 3. Thi và nhận bằng.","category":"GIAO_THONG"},
    "BANG_LAI_B2": {"code":"BANG_LAI_B2","name":"Cấp bằng lái B2","docs":"CCCD, Học lý, Giấy khám sức khỏe","coquan":"Sở GTVT","time":"20-25 ngày","lephi":"400.000đ","note":"Ô tô kinh doanh","steps":"1. Học lý thuyết, sa hình, đường trường. 2. Đăng ký thi B2. 3. Thi và nhận bằng.","category":"GIAO_THONG"},
    "BANG_LAI_DOI": {"code":"BANG_LAI_DOI","name":"Đổi bằng lái","docs":"GPLX cũ, CCCD, 1 ảnh 3x4","coquan":"Sở GTVT","time":"7-10 ngày","lephi":"135.000đ","note":"Đổi khi hết hạn","steps":"1. Chuẩn bị bằng cũ, CCCD, ảnh. 2. Đến Sở GTVT. 3. Đổi bằng mới.","category":"GIAO_THONG"},
    "BANG_LAI_CAP_LAI": {"code":"BANG_LAI_CAP_LAI","name":"Cấp lại bằng lái","docs":"CCCD, 1 ảnh 3x4, Đơn xin cấp lại","coquan":"Sở GTVT","time":"7-10 ngày","lephi":"135.000đ","note":"Khi bị mất","steps":"1. Làm đơn xin cấp lại. 2. Đến Sở GTVT. 3. Nhận bằng mới.","category":"GIAO_THONG"},

    # HỘ CHIẾU (2)
    "HO_CHIEU_CAP": {"code":"HO_CHIEU_CAP","name":"Cấp hộ chiếu","docs":"CCCD, Sổ hộ khẩu, 4 ảnh 4x6","coquan":"Phòng xuất nhập cảnh","time":"7-15 ngày","lephi":"200.000đ","note":"Hộ chiếu gắn chip 10 năm","steps":"1. Chuẩn bị CCCD, hộ khẩu, ảnh. 2. Làm đơn xin cấp hộ chiếu. 3. Nộp tại Phòng XNC. 4. Nhận hộ chiếu.","category":"HANH_CHINH"},
    "HO_CHIEU_GIA_HAN": {"code":"HO_CHIEU_GIA_HAN","name":"Gia hạn hộ chiếu","docs":"Hộ chiếu cũ, CCCD, 1 ảnh 4x6","coquan":"Phòng xuất nhập cảnh","time":"5-10 ngày","lephi":"200.000đ","note":"Trước 6 tháng hết hạn","steps":"1. Kiểm tra hạn hộ chiếu. 2. Chuẩn bị hồ sơ gia hạn. 3. Nộp tại Phòng XNC.","category":"HANH_CHINH"},

    # LÝ LỊCH (1)
    "LY_LICH_TU_PHAP": {"code":"LLTP","name":"Cấp lý lịch tư pháp","docs":"CCCD, Đơn xin cấp LLTP","coquan":"Sở Tư pháp","time":"3-5 ngày","lephi":"Miễn phí","note":"Kiểm tra án tích","steps":"1. Làm đơn xin cấp lý lịch tư pháp. 2. Nộp tại Sở Tư pháp. 3. Chờ 3-5 ngày và nhận kết quả.","category":"HANH_CHINH"},

    # NHÀ ĐẤT (2)
    "SO_DO": {"code":"SO_DO","name":"Cấp sổ đỏ (Giấy chứng nhận quyền sử dụng đất)","docs":"Đơn xin cấp, Giấy tờ về quyền sử dụng đất","coquan":"Văn phòng đăng ký đất đai","time":"10-30 ngày","lephi":"80.000đ-500.000đ","note":"Cần giấy tờ hợp pháp","steps":"1. Làm đơn xin cấp sổ đỏ. 2. Chuẩn bị giấy tờ chứng minh quyền sử dụng. 3. Nộp tại VP Đăng ký đất đai. 4. Đóng phí và nhận sổ.","category":"NHADAT"},
    "SANG_TEN": {"code":"SANG_TEN","name":"Sang tên sổ đỏ","docs":"Sổ đỏ, CCCD, Hợp đồng chuyển nhượng","coquan":"Văn phòng đăng ký đất đai","time":"10-20 ngày","lephi":"0.5% giá trị","note":"Cần công chứng hợp đồng","steps":"1. Công chứng hợp đồng chuyển nhượng. 2. Cả 2 bên đến VP Đăng ký đất đai. 3. Làm thủ tục sang tên. 4. Đóng lệ phí trước bạ.","category":"NHADAT"},

    # GIAO THÔNG XE (2)
    "DANG_KY_XE": {"code":"DANG_KY_XE","name":"Đăng ký xe ô tô","docs":"CCCD, Giấy chứng nhận quyền sở hữu xe, Giấy bảo hiểm","coquan":"Phòng đăng ký xe","time":"1-3 ngày","lephi":"1.500.000đ-3.000.000đ","note":"Trong 30 ngày kể từ khi mua","steps":"1. Chuẩn bị giấy tờ xe. 2. Mua bảo hiểm xe. 3. Đến Phòng đăng ký xe. 4. Nộp hồ sơ và nhận biển số.","category":"GIAO_THONG"},
    "CHUYEN_NHUONG_XE": {"code":"CHUYEN_NHUONG_XE","name":"Chuyển nhượng xe","docs":"CCCD cả 2 bên, Sổ đăng ký xe, Hợp đồng chuyển nhượng","coquan":"Phòng đăng ký xe","time":"2-5 ngày","lephi":"500.000đ-1.000.000đ","note":"Cả 2 cùng đến làm","steps":"1. Làm hợp đồng chuyển nhượng có công chứng. 2. Cả 2 bên đến Phòng đăng ký xe. 3. Làm thủ tục chuyển nhượng.","category":"GIAO_THONG"},

    # XÂY DỰNG (1)
    "CAP_PHEP_XAY_DUNG": {"code":"PHEP_XD","name":"Cấp phép xây dựng nhà ở","docs":"Đơn xin, Giấy chứng nhận quyền sử dụng đất, Bản vẽ quy hoạch","coquan":"Phòng Đô thị","time":"15-20 ngày","lephi":"50.000đ-200.000đ","note":"Cần quy hoạch","steps":"1. Lấy bản vẽ quy hoạch 1/500. 2. Thiết kế bản vẽ xây dựng. 3. Làm đơn xin cấp phép. 4. Nộp tại Phòng Đô thị.","category":"XAYDUNG"},

    # KINH DOANH (1)
    "CAP_PHEP_KINH_DOANH": {"code":"PHEP_KD","name":"Cấp giấy phép kinh doanh","docs":"Đơn xin cấp phép kinh doanh, CCCD, Địa chỉ kinh doanh","coquan":"Phòng Đăng ký kinh doanh hoặc UBND","time":"3-5 ngày","lephi":"Miễn phí","note":"Kinh doanh cá thể miễn phí","steps":"1. Làm đơn xin cấp phép kinh doanh. 2. Chuẩn bị CCCD, địa chỉ kinh doanh. 3. Nộp tại Phòng ĐKKD hoặc UBND.","category":"KINHDOANH"},

    # Y TẾ (2)
    "BHYT_CAP": {"code":"BHYT_CAP","name":"Cấp thẻ BHYT","docs":"CCCD, Sổ hộ khẩu","coquan":"Bảo hiểm xã hội","time":"3-5 ngày","lephi":"Miễn phí","note":"Trẻ em dưới 6 tuổi miễn phí","steps":"1. Chuẩn bị CCCD, sổ hộ khẩu. 2. Đến BHXH nơi thường trú. 3. Làm thủ tục cấp thẻ BHYT.","category":"YTE"},
    "BHYT_DOI": {"code":"BHYT_DOI","name":"Đổi thẻ BHYT","docs":"Thẻ BHYT cũ, CCCD","coquan":"Bảo hiểm xã hội","time":"1-3 ngày","lephi":"Miễn phí","note":"Đổi khi thẻ hư hỏng hoặc sai thông tin","steps":"1. Mang thẻ BHYT cũ và CCCD. 2. Đến BHXH nơi đã cấp thẻ. 3. Làm thủ tục đổi thẻ.","category":"YTE"},

    # GIÁO DỤC (1)
    "HOC_BONG": {"code":"HOC_BONG","name":"Đăng ký xét học bổng","docs":"Đơn xin học bổng, Bảng tổng kết, Học bạ","coquan":"Trường học hoặc Phòng GD-ĐT","time":"7-10 ngày","lephi":"Miễn phí","note":"Theo từng loại học bổng","steps":"1. Làm đơn xin xét học bổng. 2. Chuẩn bị hồ sơ (bảng tổng kết, học bạ). 3. Nộp tại trường hoặc Phòng GD-ĐT.","category":"GIAODUC"},

    # KHÁC (2)
    "XIN_VIEC": {"code":"XIN_VIEC","name":"Đăng ký tìm việc làm","docs":"CCCD, Hồ sơ xin việc","coquan":"Trung tâm dịch vụ việc làm","time":"1-2 ngày","lephi":"Miễn phí","note":"","steps":"1. Chuẩn bị hồ sơ xin việc (CCCD, sơ yếu lý lịch). 2. Đến Trung tâm dịch vụ việc làm. 3. Đăng ký và được giới thiệu việc làm.","category":"KHAC"},
    "TRO_CAP": {"code":"TRO_CAP","name":"Đăng ký trợ cấp xã hội","docs":"Đơn xin trợ cấp, CCCD, Sổ hộ khẩu","coquan":"UBND xã/phường","time":"7-10 ngày","lephi":"Miễn phí","note":"","steps":"1. Làm đơn xin trợ cấp xã hội. 2. Xin xác nhận hoàn cảnh khó khăn. 3. Nộp tại UBND xã/phường.","category":"KHAC"},
}

# Thêm documents từ DICHVUCONG_DATA vào vector store
for code, proc in DICHVUCONG_DATA.items():
    content = f"""THỦ TỤC: {proc['name']}
MÃ: {code}
HỒ SƠ CẦN CHUẨN BỊ: {proc['docs']}
CƠ QUAN THỰC HIỆN: {proc['coquan']}
THỜI GIAN: {proc['time']}
LỆ PHÍ: {proc['lephi']}
LƯU Ý: {proc['note']}
QUY TRÌNH: {proc['steps']}
PHÂN MỤC: {proc['category']}"""
    doc = Document(page_content=content, metadata={"code": code, "category": proc['category'], "source": "dichvucong"})
    all_docs.append(doc)

# Recreate FAISS với tất cả documents
vector_db = FAISS.from_documents(all_docs, embeddings)
print(f'   ✅ Updated FAISS: {len(all_docs)} documents (including {len(DICHVUCONG_DATA)} procedures)')
print()

# ============================================================
# PHẦN 5: BM25 & CLASSIFIER
# ============================================================

print('='*70)
print('🔍 Đang khởi tạo BM25 & Classifier...')
print('='*70)

from rank_bm25 import BM25Okapi

# BM25
tokenized_docs = [doc.page_content.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)
print('   ✅ BM25 initialized')

# Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Training data cho 50+ situations
SITUATIONS = {
    'khai_sinh_moi': ['khai sinh moi', 'sinh con', 'be moi sinh', 'dang ky khai sinh'],
    'khai_sinh_qua_han': ['khai sinh qua han', 'qua han khai sinh', 'muon khai sinh'],
    'ket_hon': ['ket hon', 'dang ky ket hon', 'dang ky cuoi', 'lay vo chong'],
    'ly_hon': ['ly hon', 'ly di', 'don ly hon', 'ly dinh'],
    'khai_tu': ['khai tu', 'bao tu', 'nguoi chet', 'dang ky khai tu'],
    'ket_hon_ngoai': ['ket hon nguoi nuoc ngoai', 'lay nguoi nuoc ngoai'],
    'cccd_cap': ['cap cccd', 'lam cccd', 'cccd lan dau', 'can cuoc cong dan'],
    'cccd_do': ['doi cccd', 'doi can cuoc', 'lam lai cccd'],
    'cccd_cap_lai': ['cap lai cccd', 'cccd bi mat', 'lam lai can cuoc'],
    'ho_khau_tach': ['tach ho khau', 'tach khau', 'tach ra khoi ho'],
    'ho_khau_nhap': ['nhap ho khau', 'nhap khau', 'vao ho khau'],
    'tam_tru': ['tam tru', 'dang ky tam tru', 'xin tam tru'],
    'tru_so': ['tru so', 'tru qua han', 'dang ky tru quan'],
    'tam_vang': ['tam vang', 'dang ky tam vang', 'bao tam vang'],
    'bang_lai_a1': ['bang lai a1', 'bang lai xe may', 'lai xe may'],
    'bang_lai_a2': ['bang lai a2', 'bang lai xe cong nong'],
    'bang_lai_b1': ['bang lai b1', 'bang lai oto duoi 9 cho'],
    'bang_lai_b2': ['bang lai b2', 'bang lai oto', 'lai xe o to'],
    'bang_lai_do': ['doi bang lai', 'lam lai bang lai', 'bang lai mat'],
    'bang_lai_cap_lai': ['cap lai bang lai', 'bang lai bi mat'],
    'ho_chieu_cap': ['lam ho chieu', 'cap ho chieu', 'passport'],
    'ho_chieu_gia_han': ['gia han ho chieu', 'doi ho chieu'],
    'ly_lich_tu_phap': ['ly lich tu phap', 'an tien an', 'so tu phap'],
    'so_do': ['so do', 'giay chu quyen', 'sodo', 'cap so do nha'],
    'sang_ten': ['sang ten', 'chuyen nhuong', 'chuyen ten'],
    'kinh_doanh': ['kinh doanh', 'mo cua hang', 'giay phep kinh doanh'],
    'dang_ky_xe': ['dang ky xe', 'dang ki xe o to', 'nghia vu xe'],
    'chuyen_nhuong_xe': ['chuyen nhuong xe', 'ban xe', 'mua xe'],
    'xay_dung': ['xay dung', 'xay nha', 'phep xay dung'],
    'bhyt_cap': ['cap the bhyt', 'lam bao hiem y te'],
    'bhyt_do': ['doi the bhyt', 'gia han bao hiem y te'],
    'hoc_bong': ['dang ky hoc bong', 'xin hoc bong'],
    'xin_viec': ['dang ky tim viec lam', 'tim viec'],
    'tro_cap': ['dang ky tro cap xa hoi', 'tro cap'],
    'hoi': ['hoi', 'hoi ve', 'tu van', 'giup doi'],
}

# Train classifier
train_text = []
train_label = []
for lbl, exs in SITUATIONS.items():
    for ex in exs:
        train_text.append(ex)
        train_label.append(lbl)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X = vec.fit_transform(train_text)

clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft'
)
clf.fit(X, train_label)
print(f'   ✅ Classifier trained: {len(SITUATIONS)} situations')
print()

# ============================================================
# PHẦN 6: LOAD LLM (QWEN 2.5 3B)
# ============================================================

print('='*70)
print('🤖 Đang tải LLM Qwen 2.5 3B...')
print('='*70)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map='auto',
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Create pipeline
llm = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
    repetition_penalty=1.1
)
print('   ✅ LLM loaded successfully!')
print()

# ============================================================
# PHẦN 7: VOICE AI (WHISPER + EDGE-TTS)
# ============================================================

print('='*70)
print('🎤 Đang tải Voice AI...')
print('='*70)

import whisper
import edge_tts

# Whisper model
whisper_model = whisper.load_model('base')
print('   ✅ Whisper loaded (base model)')

# Edge-TTS sẽ dùng trực tiếp trong hàm
print('   ✅ Edge-TTS ready')
print()

# ============================================================
# PHẦN 8: AI PERSONAS & QUERY PATTERNS
# ============================================================

AI_PERSONAS = {
    "thuong": {
        "name": "Chị Thuong - Cán bộ tư vấn thân thiện",
        "tone": "thân thiện, gần gũi, dùng từ ngữ bình dân",
        "greeting": ["Chào bác/cháu ạ,", "Dạ vâng,", "Bác/cháu hỏi đúng người rồi ạ,"],
        "closing": ["Có gì bác/chứ cứ hỏi thêm nhé!", "Chúc bác/cháu làm thủ tục thuận lợi ạ!"],
        "style": "conversational"
    },
    "chuyen": {
        "name": "Anh Chuyen - Chuyên viên hành chính công",
        "tone": "chuyên nghiệp, ngắn gọn, súc tích",
        "greeting": ["Dạ, về vấn đề", "Theo quy định,", "Về thủ tục"],
        "closing": ["Trân trọng!", "Thông tin trên để tham khảo ạ."],
        "style": "professional"
    },
    "chi_tiet": {
        "name": "Cô Chi Tiet - Cán bộ hướng dẫn chi tiết",
        "tone": "chi tiết, từng bước, cẩn thận",
        "greeting": ["Dạ để em hướng dẫn chi tiết cho bác/cháu ạ,", "Em sẽ giải thích kỹ càng ạ,"],
        "closing": ["Bác/cháu làm theo như trên là được ạ!", "Nếu chưa hiểu chỗ nào cứ hỏi thêm ạ!"],
        "style": "detailed"
    }
}

QUERY_PATTERNS = {
    "can_gi": ["cần gì", "giấy tờ gì", "chuẩn bị gì", "hồ sơ gì"],
    "o_day": ["ở đâu", "nơi nào", "địa chỉ", "đến đâu"],
    "bao_lau": ["bao lâu", "mất bao lâu", "thời gian", "how long"],
    "bao_nhieu": ["bao nhiêu", "bao tiền", "phí", "lệ phí", "cost"],
    "the_quy_trinh": ["quy trình", "cách làm", "làm thế nào", "làm sao"],
    "nguoi_nuoc_ngoai": ["người nước ngoài", "ngoài", "không phải VN"],
    "qua_han": ["qua hạn", "quá hạn", "muộn"],
    "mat": ["mất", "thất lạc", "đánh rơi"],
    "doi": ["đổi", "làm lại", "cập nhật"],
    "moi": ["lần đầu", "mới"],
}

# ============================================================
# PHẦN 9: CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.context = {}
        self.max_history = max_history
        self.persona = "thuong"

    def add_message(self, role: str, content: str, metadata: dict = None):
        self.history.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def update_context(self, key: str, value: any):
        self.context[key] = value

    def get_context_summary(self) -> str:
        if not self.context:
            return ""
        parts = []
        if "situation" in self.context:
            parts.append(f"Vấn đề: {self.context['situation']}")
        return " | ".join(parts)

    def get_relevant_history(self, current_query: str) -> str:
        if not self.history:
            return ""
        recent = self.history[-6:]
        relevant = []
        for msg in recent:
            if msg["role"] == "user":
                relevant.append(f"Người dùng: {msg['content']}")
        return "\n".join(relevant)

    def set_persona(self, persona: str):
        if persona in AI_PERSONAS:
            self.persona = persona

    def get_persona(self) -> dict:
        return AI_PERSONAS.get(self.persona, AI_PERSONAS["thuong"])

    def detect_situation_from_history(self) -> str:
        for msg in reversed(self.history):
            if msg.get("metadata", {}).get("situation"):
                return msg["metadata"]["situation"]
        return ""

    def clear(self):
        self.history = []
        self.context = {}

conversation_memory = ConversationMemory()

# ============================================================
# PHẦN 10: HELPER FUNCTIONS
# ============================================================

def get_diverse_greeting(persona_key: str) -> str:
    persona = AI_PERSONAS.get(persona_key, AI_PERSONAS["thuong"])
    return random.choice(persona['greeting'])

def get_diverse_closing(persona_key: str) -> str:
    persona = AI_PERSONAS.get(persona_key, AI_PERSONAS["thuong"])
    return random.choice(persona['closing'])

def detect_intent(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    intents = {k: False for k in QUERY_PATTERNS.keys()}
    for k, patterns in QUERY_PATTERNS.items():
        if any(p in query_lower for p in patterns):
            intents[k] = True
    return intents

def expand_query(query: str) -> List[str]:
    expanded = [query]
    query_lower = query.lower()
    for pattern, keywords in QUERY_PATTERNS.items():
        for kw in keywords:
            if kw in query_lower:
                for syn in [kw + " di", kw + " the"]:
                    if syn not in query_lower:
                        expanded.append(query.replace(kw, syn))
    return list(set(expanded))

def build_chain_of_thought_prompt(query: str, situation: str, context: str,
                                   retrieved_docs: list, intents: dict) -> str:
    persona = conversation_memory.get_persona()
    step1 = f"""BƯỚC 1 - XÁC ĐỊNH VẤN ĐỀ:
Người dùng đang hỏi về: {situation.replace('_', ' ').upper()}
Câu hỏi cụ thể: "{query}"
"""
    intent_list = [k for k, v in intents.items() if v]
    step2 = f"""BƯỚC 2 - PHÂN TÍCH Ý ĐỊNH:
Người dùng muốn biết: {', '.join(intent_list) if intent_list else 'thông tin chung'}
"""
    step3 = """BƯỚC 3 - TRA CỨU THÔNG TIN:
Dựa trên tài liệu và quy định hiện hành:
"""
    for i, doc in enumerate(retrieved_docs[:3], 1):
        step3 += f"\n[{i}] {doc.page_content.strip()[:200]}..."
    step4 = f"""BƯỚC 4 - TỔNG HỢP TRẢ LỜI:
Phong cách: {persona['tone']}
"""
    return step1 + step2 + step3 + step4

def calculate_temperature(query: str, situation: str) -> float:
    simple_patterns = ["cần gì", "ở đâu", "bao lâu", "bao nhiêu"]
    for pattern in simple_patterns:
        if pattern in query.lower():
            return 0.1
    return 0.2

def build_smart_response(query: str, situation: str, docs: list, intents: dict, persona_key: str = "thuong") -> str:
    persona = AI_PERSONAS.get(persona_key, AI_PERSONAS["thuong"])
    parts = [get_diverse_greeting(persona_key)]

    proc_data = None
    for code, proc in DICHVUCONG_DATA.items():
        if situation.replace("_", "").upper() in code:
            proc_data = proc
            break

    if proc_data:
        parts.append(f"\nVề **{proc_data['name']}**, em xin tư vấn:")
        parts.append(f"\n📋 **Giấy tờ:** {proc_data['docs']}")
        parts.append(f"\n🏢 **Nơi làm:** {proc_data['coquan']}")
        parts.append(f"\n⏰ **Thời gian:** {proc_data['time']}")
        parts.append(f"\n💰 **Lệ phí:** {proc_data['lephi']}")
        if proc_data['note']:
            parts.append(f"\n⚠️ **Lưu ý:** {proc_data['note']}")
        parts.append(f"\n📝 **Quy trình:** {proc_data['steps']}")
    else:
        parts.append("\nvề câu hỏi này, em xin phép tư vấn:")
        for i, doc in enumerate(docs[:2], 1):
            parts.append(f"\nDựa trên tài liệu [{i}]: {doc.page_content[:150]}...")

    parts.append(f"\n\n{get_diverse_closing(persona_key)}")
    return "\n".join(parts)

# ============================================================
# PHẦN 11: RAG SEARCH FUNCTIONS
# ============================================================

def hybrid_search(query: str, k_final: int = 5):
    dense_results = vector_db.similarity_search_with_score(query, k=k_final*2)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:k_final*2]

    combined = {}
    for doc, score in dense_results:
        doc_id = doc.page_content[:100]
        combined[doc_id] = {'doc': doc, 'score': 1 - score}

    for idx in top_indices:
        if idx < len(all_docs):
            doc_id = all_docs[idx].page_content[:100]
            if doc_id not in combined:
                combined[doc_id] = {'doc': all_docs[idx], 'score': 0.5}

    sorted_docs = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:k_final]
    return [item['doc'] for item in sorted_docs]

def advanced_hybrid_search(query: str, k_final: int = 5):
    expanded_queries = expand_query(query)
    all_results = {}
    for expanded_q in expanded_queries[:3]:
        docs = hybrid_search(expanded_q, k_final=8)
        for doc in docs:
            doc_id = doc.page_content[:50]
            if doc_id not in all_results:
                all_results[doc_id] = {"doc": doc, "score": 1.0}
            else:
                all_results[doc_id]["score"] += 0.5

    query_words = set(query.lower().split())
    for doc_id, item in all_results.items():
        doc_words = set(item["doc"].page_content.lower().split())
        overlap = len(query_words & doc_words)
        item["score"] += overlap * 0.1

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)
    return [item[1]["doc"] for item in sorted_results[:k_final]]

# ============================================================
# PHẦN 12: AI PROCESS FUNCTIONS
# ============================================================

def ai_process(query: str) -> dict:
    try:
        start_time = time.time()
        intents = detect_intent(query)

        try:
            situation = clf.predict(vec.transform([query]))[0]
        except:
            situation = 'hoi'

        docs = advanced_hybrid_search(query, k_final=5)

        proc_info = ""
        for code, proc in DICHVUCONG_DATA.items():
            if situation.replace("_", "").upper() in code:
                proc = proc
                proc_info = f"THỦ TỤC: {proc['name']}\nHỒ SƠ: {proc['docs']}\nCƠ QUAN: {proc['coquan']}"
                break

        response = build_smart_response(query, situation, docs, intents, "thuong")

        return {
            "response": response,
            "docs_count": len(docs),
            "process_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        return {
            "response": f"Lỗi: {str(e)}",
            "docs_count": 0,
            "process_time": 0
        }

def ai_process_pro(query: str, use_memory: bool = True) -> dict:
    try:
        start_time = time.time()

        torch.cuda.empty_cache()
        gc.collect()

        intents = detect_intent(query)

        try:
            situation = clf.predict(vec.transform([query]))[0]
        except:
            situation = 'hoi'

        docs = advanced_hybrid_search(query, k_final=5)

        proc_info = ""
        proc_data = None
        for code, proc in DICHVUCONG_DATA.items():
            if situation.replace("_", "").upper() in code:
                proc_data = proc
                proc_info = f"""THỦ TỤC: {proc['name']}
HỒ SƠ: {proc['docs']}
CƠ QUAN: {proc['coquan']}
THỜI GIAN: {proc['time']}
LỆ PHÍ: {proc['lephi']}
LƯU Ý: {proc['note']}
QUY TRÌNH: {proc['steps']}"""
                break

        context_summary = ""
        relevant_history = ""
        if use_memory:
            conversation_memory.update_context("situation", situation)
            context_summary = conversation_memory.get_context_summary()
            relevant_history = conversation_memory.get_relevant_history(query)

        temperature = calculate_temperature(query, situation)

        cot_prompt = build_chain_of_thought_prompt(
            query=query,
            situation=situation,
            context=context_summary,
            retrieved_docs=docs,
            intents=intents
        )

        full_prompt = f"""Bạn là cán bộ tư vấn thủ tục hành chính công tại UBND.

{cot_prompt}

CÂU HỎI: {query}

{proc_info}

LỊCH SỬ ĐỌC CHUẨN KHI CÓ:
{relevant_history}

HÃY TRẢ LỜI NGƯỜI DÂNG:"""

        messages = [
            {"role": "system", "content": "Bạn là cán bộ UBND tận tâm."},
            {"role": "user", "content": full_prompt}
        ]

        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm(
            text_prompt,
            max_new_tokens=768,
            temperature=temperature,
            repetition_penalty=1.15,
            do_sample=temperature > 0.2
        )

        response_text = outputs[0]["generated_text"]

        if "<|im_start|>assistant\n" in response_text:
            response_text = response_text.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response_text:
            response_text = response_text.split("<|im_end|>")[0]

        llm_response = response_text.strip()

        clean_lines = []
        for line in llm_response.split('\n'):
            if any('\u4e00' <= char <= '\u9fff' for char in line):
                continue
            clean_lines.append(line)

        final_response = '\n'.join(clean_lines).strip()

        if len(final_response) < 50:
            final_response = build_smart_response(query, situation, docs, intents, "thuong")

        if len(final_response) < 30:
            final_response = "Dạ bác/cháu hỏi đúng người rồi ạ! Hiện tại em chưa có đầy đủ thông tin."

        if use_memory:
            conversation_memory.add_message("user", query, {"situation": situation})
            conversation_memory.add_message("assistant", final_response, {"situation": situation})

        torch.cuda.empty_cache()
        gc.collect()

        process_time = time.time() - start_time

        return {
            "situation": situation,
            "response": final_response,
            "docs_count": len(docs),
            "process_time": round(process_time, 2),
            "intents": intents,
            "temperature": temperature
        }

    except Exception as e:
        print(f'AI Process PRO Error: {e}')
        traceback.print_exc()

        try:
            fallback = build_smart_response(
                query,
                conversation_memory.detect_situation_from_history() or 'hoi',
                [],
                detect_intent(query),
                ""
            )
            return {
                "situation": "fallback",
                "response": fallback,
                "docs_count": 0,
                "process_time": 0.5
            }
        except:
            return {
                "situation": "error",
                "response": "Dạ xin lỗi, hệ thống đang bận.",
                "docs_count": 0,
                "process_time": 0
            }

# ============================================================
# PHẦN 13: VOICE FUNCTIONS
# ============================================================

def speech_to_text(audio_path: str) -> str:
    try:
        if audio_path and os.path.exists(audio_path):
            result = whisper_model.transcribe(audio_path, language='vi', fp16=False)
            return result["text"].strip()
    except Exception as e:
        print(f"Whisper error: {e}")
    return ""

async def text_to_speech(text: str, voice: str = 'vi-VN-HoaiMyNeural') -> str:
    try:
        import edge_tts
        text = str(text)[:1500].replace('*','').replace('#','').replace('_','')
        timestamp = datetime.now().strftime('%H%M%S')
        output_path = os.path.join('./audio_output', f"audio_{timestamp}.mp3")

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ============================================================
# PHẦN 14: GRADIO INTERFACE
# ============================================================

import gradio as gr

# Basic chat function
def process_chat(audio, text, history):
    try:
        if history is None:
            history = []

        query = ''
        if audio and not os.path.isdir(audio):
            query = speech_to_text(audio)
        elif text and text.strip():
            query = text.strip()

        if len(query) < 3:
            return history, None, ''

        result = ai_process(query)
        response = f"**🏛️ AI Hành Chính Công**\n\n{result['response']}"

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        return history, None, ''
    except Exception as e:
        return history, None, f"Lỗi: {str(e)}"

def clear_chat():
    return [], None, ''

# PRO chat function
def process_chat_pro(audio, text, history, persona="thuong"):
    try:
        if history is None:
            history = []

        conversation_memory.set_persona(persona)

        query = ''
        if audio and not os.path.isdir(audio):
            query = speech_to_text(audio)
        elif text and text.strip():
            query = text.strip()

        if len(query) < 3:
            return history, None, ''

        result = ai_process_pro(query, use_memory=True)

        response = f"""**🏛️ AI HÀNH CHÍNH CÔNG VIP PRO**

**📋 Tình huống:** `{result['situation']}`

{result['response']}

---
_⏱️ Xử lý: {result['process_time']}s | 📄 Tài liệu: {result['docs_count']} | 🌡️ Temp: {result.get('temperature', 0.2):.2f}
"""

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        audio_out = None
        try:
            audio_out = loop.run_until_complete(text_to_speech(result['response']))
        except:
            pass

        return history, audio_out, ''
    except Exception as e:
        import traceback
        traceback.print_exc()
        return history, None, f"Lỗi: {str(e)}"

def clear_chat_pro():
    conversation_memory.clear()
    return [], None, ''

def change_persona(new_persona):
    conversation_memory.set_persona(new_persona)
    persona_names = {
        "thuong": "Chị Thuong - Thân thiện",
        "chuyen": "Anh Chuyen - Chuyên nghiệp",
        "chi_tiet": "Cô Chi Tiet - Chi tiết"
    }
    return f"✅ Đã chuyển sang: {persona_names.get(new_persona, new_persona)}"

# Admin function
def admin_add(code, name, docs, coquan, time_str, lephi, note, steps):
    try:
        DICHVUCONG_DATA[code] = {
            "code": code, "name": name, "docs": docs, "coquan": coquan,
            "time": time_str, "lephi": lephi, "note": note, "steps": steps,
            "category": "CUSTOM"
        }
        return f"✅ Đã thêm thủ tục: {name}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

def get_procedures_list():
    return "\n".join([f"{code}: {proc['name']}" for code, proc in DICHVUCONG_DATA.items()])

# CSS
pro_css = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.chatbot { border-radius: 20px; }
.pro-badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
"""

# Build PRO interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue='violet', secondary_hue='purple'), css=pro_css) as demo_pro:
    gr.HTML("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; font-size: 2.8em;">🏛️ AI HÀNH CHÍNH CÔNG</h1>
        <p style="color: white; margin: 15px 0 0 0; font-size: 1.2em;">Hệ thống tư vấn thủ tục hành chính công thông minh</p>
        <div style="margin-top: 15px;">
            <span class="pro-badge">37+ Thủ tục</span>
            <span class="pro-badge">Hybrid RAG</span>
            <span class="pro-badge">Voice AI</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # Chat Tab
        with gr.Tab("💬 Chat AI"):
            with gr.Row():
                with gr.Column(scale=2):
                    chat = gr.Chatbot(height=500, type='messages', label='💬 Lịch sử chat')
                    audio_out = gr.Audio(label='🔊 Nghe câu trả lời', autoplay=True)

                with gr.Column(scale=1):
                    persona_selector = gr.Radio(
                        choices=[
                            ("👩‍💼 Chị Thuong - Thân thiện, gần gũi", "thuong"),
                            ("👨‍💼 Anh Chuyen - Chuyên nghiệp, ngắn gọn", "chuyen"),
                            ("👩‍🏫 Cô Chi Tiet - Chi tiết, từng bước", "chi_tiet")
                        ],
                        value="thuong",
                        label="🎭 Phong cách tư vấn:",
                        interactive=True
                    )
                    persona_status = gr.Textbox(label="Trạng thái", interactive=False)

                    gr.HTML("<br>")

                    audio_in = gr.Audio(sources=['microphone'], type='filepath', label='🎤 Nói câu hỏi')
                    msg_in = gr.Textbox(label='✏️ Hoặc gõ câu hỏi', lines=4, placeholder='Ví dụ: Làm bằng lái xe máy cần giấy tờ gì?')

                    with gr.Row():
                        btn_send = gr.Button('🚀 GỬI', variant='primary', size='lg')
                        btn_clear = gr.Button('🔄 LÀM MỚI', size='lg')

            gr.Examples(
                examples=[
                    [None, 'Làm bằng lái xe máy cần giấy tờ gì?', 'thuong'],
                    [None, 'Khai sinh quá hạn phải làm sao?', 'chi_tiet'],
                    [None, 'Đăng ký xe ô tô mới ở đâu?', 'chuyen'],
                    [None, 'Tách hộ khẩu cần những gì?', 'thuong'],
                ],
                inputs=[audio_in, msg_in, persona_selector]
            )

            btn_send.click(process_chat_pro, [audio_in, msg_in, chat, persona_selector], [chat, audio_out, msg_in])
            msg_in.submit(process_chat_pro, [audio_in, msg_in, chat, persona_selector], [chat, audio_out, msg_in])
            btn_clear.click(clear_chat_pro, [chat, audio_out, msg_in])
            persona_selector.change(change_persona, [persona_selector], [persona_status])

        # Admin Tab
        with gr.Tab("⚙️ Admin Panel"):
            gr.Markdown("### 📝 Thêm thủ tục mới")

            with gr.Row():
                with gr.Column():
                    admin_code = gr.Textbox(label="Mã thủ tục", placeholder="VD: NEW_PROC")
                    admin_name = gr.Textbox(label="Tên thủ tục", placeholder="VD: Thủ tục mới")
                    admin_docs = gr.Textbox(label="Giấy tờ cần", placeholder="CCCD, Sổ hộ khẩu...")
                    admin_coquan = gr.Textbox(label="Cơ quan", placeholder="UBND xã/phường")
                with gr.Column():
                    admin_time = gr.Textbox(label="Thời gian", placeholder="1-3 ngày")
                    admin_lephi = gr.Textbox(label="Lệ phí", placeholder="Miễn phí")
                    admin_note = gr.Textbox(label="Lưu ý", placeholder="Ghi chú")
                    admin_steps = gr.Textbox(label="Quy trình", placeholder="1. ..., 2. ...", lines=3)

            btn_add = gr.Button("➕ THÊM THỦ TỤC", variant="primary")
            admin_output = gr.Textbox(label="Kết quả", interactive=False)

            gr.Markdown("### 📋 Danh sách thủ tục")
            admin_list = gr.Textbox(label="Tất cả thủ tục", lines=8, interactive=False)
            btn_refresh = gr.Button("🔄 Làm mới danh sách")

            btn_add.click(admin_add, [admin_code, admin_name, admin_docs, admin_coquan, admin_time, admin_lephi, admin_note, admin_steps], [admin_output])
            btn_refresh.click(get_procedures_list, [admin_list])

# ============================================================
# PHẦN 15: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print('='*70)
    print('🚀 KHỞI ĐỘNG ỨNG DỤNG AI HÀNH CHÍNH CÔNG')
    print('='*70)
    print()
    print('📊 Thống kê:')
    print(f'   - Dataset: {len(df1) + len(df2):,} samples + {len(DICHVUCONG_DATA)} thủ tục')
    print(f'   - Situations: {len(SITUATIONS)}')
    print(f'   - Personas: {len(AI_PERSONAS)}')
    print(f'   - Model: Qwen 2.5 3B')
    print(f'   - RAG: Hybrid + Re-ranking')
    print(f'   - Voice: Whisper + Edge-TTS')
    print()
    print('='*70)
    print('  Đang chạy Gradio interface...')
    print('  Uygul dụng sẽ có sẵn tại: http://localhost:7860')
    print('='*70)
    print()

    demo_pro.launch(
        share=False,
        server_name='0.0.0.0',
        server_port=7860,
        show_error=True
    )
