# Sơ đồ Hệ thống Chatbot Y tế - Dành cho Khách hàng

## Tổng quan Hệ thống

Chatbot Y tế thông minh phục vụ **2 chuyên khoa chính**:
- **Răng Hàm Mặt (RHM)**
- **Nội Tiết (Đái tháo đường)**

## 👥 Đối tượng Người dùng

```mermaid
graph LR
    subgraph "NGƯỜI DÙNG"
        BN1[Bệnh nhân<br/>Răng Hàm Mặt]
        BN2[Bệnh nhân<br/>Đái tháo đường]
        BS1[Bác sĩ<br/>Răng Hàm Mặt]
        BS2[Bác sĩ<br/>Nội tiết]
    end
    
    BN1 --> CHATBOT[CHATBOT Y TẾ]
    BN2 --> CHATBOT
    BS1 --> CHATBOT
    BS2 --> CHATBOT
    
    CHATBOT --> TL[Tư vấn phù hợp<br/>theo từng đối tượng]
```

## Quy trình Hoạt động

```mermaid
flowchart TD
    START([Khách hàng bắt đầu]) --> CHON[Chọn vai trò<br/>Bệnh nhân/Bác sĩ + Chuyên khoa]
    
    CHON --> HOI[Đặt câu hỏi]
    
    HOI --> AI{AI phân tích<br/>câu hỏi}
    
    AI -->|Câu hỏi y tế| KB[Tìm kiếm<br/>Cơ sở tri thức]
    AI -->|Chào hỏi| CHAO[Chào hỏi thân thiện]
    AI -->|Cần gợi ý| GOI[Gợi ý chủ đề]
    
    KB --> DANH{Đánh giá<br/>độ chính xác}
    
    DANH -->|Cao| TL[AI trả lời chi tiết<br/>+ Gợi ý câu hỏi]
    DANH -->|Thấp| LAMRO[Yêu cầu làm rõ<br/>+ Gợi ý câu hỏi]
    
    CHAO --> GOI
    GOI --> END([Kết thúc])
    TL --> END
    LAMRO --> END
    
    style START fill:#e1f5fe
    style END fill:#f3e5f5
    style AI fill:#fff3e0
    style TL fill:#e8f5e8
```

## Tính năng Chính

### **Cá nhân hóa theo Vai trò**
- **Bệnh nhân**: Ngôn ngữ dễ hiểu, lời khuyên cơ bản
- **Bác sĩ**: Thông tin chuyên môn, hỗ trợ chẩn đoán

### **AI Thông minh**
- Hiểu ý định câu hỏi
- Tìm kiếm thông tin chính xác
- Đưa ra câu trả lời phù hợp

### **Cơ sở Tri thức Phong phú**
- Hàng nghìn câu hỏi - đáp án
- Được cập nhật thường xuyên
- Phân loại theo chuyên khoa

### **Phản hồi Nhanh chóng**
- Thời gian phản hồi < 3 giây
- Hoạt động 24/7
- Giao diện thân thiện

## Giao diện Người dùng

```mermaid
graph TB
    subgraph "UI"
        HEADER[CHATBOT Y TẾ<br/>Tư vấn Răng Hàm Mặt & Nội Tiết]
        
        subgraph "CHỌN VAI TRÒ"
            ROLE1[Bệnh nhân RHM]
            ROLE2[Bệnh nhân ĐTĐ]
            ROLE3[Bác sĩ RHM]
            ROLE4[Bác sĩ Nội tiết]
        end
        
        subgraph "CHAT"
            MESSAGES[Tin nhắn]
            INPUT[Nhập câu hỏi...]
            SEND[Gửi]
        end
        
        subgraph "GỢI Ý"
            SUG1[Thuốc điều trị sâu răng?]
            SUG2[Chế độ ăn cho người tiểu đường?]
            SUG3[Cách vệ sinh răng miệng?]
        end
    end
    
    HEADER --> ROLE1
    ROLE1 --> MESSAGES
    MESSAGES --> INPUT
    INPUT --> SEND
    SEND --> SUG1
```

## Lợi ích cho Khách hàng

### **Cho Bệnh nhân**
- Tư vấn y tế 24/7
- Thông tin dễ hiểu, đáng tin cậy  
- Gợi ý câu hỏi hữu ích
- Không cần chờ đợi

### **Cho Bác sĩ**
- Hỗ trợ tra cứu nhanh
- Thông tin chuyên môn cập nhật
- Tiết kiệm thời gian
- Hỗ trợ quyết định lâm sàng

## Đảm bảo Chất lượng

### **Độ chính xác cao**
- Cơ sở tri thức được kiểm duyệt bởi chuyên gia
- AI được huấn luyện với dữ liệu y tế chất lượng
- Hệ thống đánh giá độ tin cậy

### **An toàn thông tin**
- Không lưu trữ thông tin cá nhân nhạy cảm
- Tuân thủ quy định bảo mật y tế
- Mã hóa dữ liệu truyền tải

### **Cải tiến liên tục**
- Thu thập phản hồi người dùng
- Cập nhật kiến thức thường xuyên
- Tối ưu hiệu suất

## Roadmap Phát triển

```mermaid
timeline
    title Lộ trình Phát triển Chatbot Y tế
    
    section Giai đoạn 1
        Hiện tại : Chatbot cơ bản
                 : 2 chuyên khoa RHM & Nội tiết
                 : Tư vấn theo vai trò
    
    section Giai đoạn 2
        3-6 tháng : Thêm chuyên khoa mới
                  : Tích hợp voice chat
                  : Mobile app
    
    section Giai đoạn 3
        6-12 tháng : AI nâng cao
                   : Phân tích hình ảnh y tế
                   : Tích hợp hồ sơ bệnh án
```

## Liên hệ & Hỗ trợ

**Đơn vị phát triển**: Đội ngũ AI Y tế  
**Email hỗ trợ**: support@chatbot-yte.com  
**Hotline**: 1900-xxxx  
**Website**: www.chatbot-yte.com  

---

*Chatbot Y tế - Đồng hành cùng sức khỏe của bạn 24/7*
