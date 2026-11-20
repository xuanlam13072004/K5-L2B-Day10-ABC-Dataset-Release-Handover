# Bảo mật khi làm việc
> Trả lời gắn với lô bạn vừa bàn giao, không chép slide. Mỗi mục vài câu.
## 1. Ca USB: hết giờ, muốn chép lô về nhà làm tiếp
Không được chép dữ liệu lô `lab250_79c4d088f4c4` hay `lab250_79c4d088f4c4_v2` ra USB cá nhân để mang về nhà vì dữ liệu thuộc phạm vi kiểm soát nội bộ, dễ gây rò rỉ hoặc thất lạc khi mang ra ngoài môi trường làm việc. Việc gửi file qua email cá nhân hoặc nhờ người khác làm hộ cũng vi phạm nguyên tắc bảo mật do làm dữ liệu rời khỏi hệ thống phân quyền và để người không có thẩm quyền tiếp cận. Thay vào đó, cần báo cáo quản lý/lead để xin cấp quyền truy cập từ xa qua VPN/máy trạm công ty hoặc đăng ký làm thêm giờ tại văn phòng trên tài khoản được cấp.
## 2. Năm việc: được hay không
- **Mở job được giao trên máy công ty:** **Được**, vì đây là thao tác đúng phạm vi công việc, đúng người được phân công trên hạ tầng bảo mật của tổ chức.
- **Chụp màn hình gửi người quen:** **Không được**, vì ảnh dữ liệu (kể cả ảnh COCO hay dữ liệu khách hàng) thuộc tài sản dự án, không được phát tán ra bên ngoài.
- **Chép lô ra USB:** **Không được**, vì thiết bị lưu trữ cá nhân không được kiểm soát mã hóa/thu hồi, dễ gây lộ lọt cả lô dữ liệu.
- **Dán nhãn thật lên chatbot:** **Không được**, vì dữ liệu nội bộ khi đưa lên dịch vụ AI bên ngoài có nguy cơ bị lưu trữ và dùng để huấn luyện lại.
- **Đưa tài khoản CVAT cho đồng nghiệp:** **Không được**, vì mỗi tài khoản gắn với trách nhiệm truy vết (annotator/reviewer) trong `jobs.csv`, dùng chung làm sai lệch lịch sử thao tác và vi phạm phân quyền.
## 3. Ca khuôn mặt: thấy mặt người rõ trong lô, guideline yêu cầu làm mờ
Khi phát hiện ảnh trong lô `lab250_79c4d088f4c4_v2` có khuôn mặt người nhìn rõ (ví dụ các ảnh có nhãn `person` như `000000000785, 000000000872, 000000000885`) trong khi guideline yêu cầu làm mờ khuôn mặt, phải **dừng bàn giao ngay**, đặt trạng thái lô là **Hold** và báo ngay cho **người phụ trách QC / quản lý dự án (PM)** để xử lý theo quy trình. Tuyệt đối không tự ý dùng phần mềm chỉnh sửa làm mờ hay tự xóa ảnh khỏi lô vì làm thay đổi mã băm `sha256` trong `manifest.csv`, sai lệch số lượng ảnh/nhãn so với `label_summary.csv` và mất dấu vết dữ liệu gốc trước khi có quyết định chính thức.
