Release: lab250_29f82c6d3554_v2
Parent: lab250_29f82c6d3554
Số ảnh: 244
Số nhãn: 1669
Guideline: v1.2 (J01-J03), v1.3 (J04-J10)
QC v1: 14/250 = 5,6%
Kết luận v1: Hold — Tỉ lệ lỗi 5,6% vượt ngưỡng 2% (đã ghi HOLD và kiểm tra impact R-021)
Kết luận v2: Rework/Handover v2 — Đã tách 6 ảnh lỗi sang rework_queue.csv, đồng nhất split scene_818 về train, cập nhật hash thực tế cho 2 ảnh wrong_hash và chuẩn hóa guideline theo jobs.csv; còn 3 ảnh lỗi panoptic giữ nguyên ở v2 để báo team panoptic
Ghi chú: scene_818 đưa 000000007977 từ val về train; hash đã mở kiểm tra ảnh 000000011760 và 000000024144 (file ảnh hợp lệ, cập nhật sha256 thực tế trong manifest.csv); guideline chuẩn theo jobs.csv; 6 ảnh rework giao lại đúng annotator/reviewer trong rework_queue.csv (A03/Q02, A02/Q02, A03/Q01, A04/Q03, A04/Q02); 3 ảnh lỗi panoptic ở lại v2 báo team panoptic: 000000009590 (invalid_category_id), 000000016249 và 000000023781 (thing_overlap).
