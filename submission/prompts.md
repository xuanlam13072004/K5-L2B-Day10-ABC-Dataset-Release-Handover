# Dùng AI

| # | Ai dùng | Công cụ | Prompt (tóm tắt) | AI sai ở đâu | Nhóm kiểm lại thế nào |
|---|---|---|---|---|---|
| 1 | Nguyễn Bình Anh | Cursor / Gemini | Viết `validator/candidates.py` chỉ dùng thư viện chuẩn và `pngutil.py` để đối chiếu Detection (YOLO) với Panoptic mask và lọc 5 loại lỗi (`class_swap`, `category_swap`, `bbox_drift`, `missing_box`, `instance_merge`). | Bản đầu chỉ viết hàm `find_candidates(delivery_dir)` nên khi ghép vào `validator/validate.py` bị báo lỗi `AttributeError: module 'candidates' has no attribute 'build'`. | Tự kiểm tra giao diện `validate.py`, bổ sung hàm `build(L, skip_images)` kèm lọc bỏ `skip_images`; chạy thực nghiệm ra **4 ứng viên** trên lô sạch (`runs/clean`) và **16 ứng viên** trên lô nhóm (`runs/r9007`). |
