# 📷 อ่านข้อความจากภาพและไฟล์ (OCR)

สคริปต์สำหรับอ่านข้อความจากภาพและไฟล์ text ในโฟลเดอร์ Book แล้วบันทึกลง output.txt

## ⚡ วิธีติดตั้ง (เพียง 2 ขั้นตอน)

### 1. ติดตั้ง Tesseract OCR
```bash
brew install tesseract tesseract-lang
```

### 2. ติดตั้ง Python Libraries
```bash
pip3 install pillow pytesseract opencv-python numpy
```

## 🚀 วิธีใช้งาน

### อ่านไฟล์จากโฟลเดอร์ Book (ใช้อันนี้)
```bash
python3 read_book_to_text.py
```
ผลลัพธ์จะถูกบันทึกใน `output.txt`

### อ่านไฟล์จากโฟลเดอร์ปัจจุบัน
```bash
python3 extract_text_from_images.py
```
ผลลัพธ์จะถูกบันทึกใน `sum.txt`

## 📁 โครงสร้างไฟล์

```
.
├── read_book_to_text.py          # อ่านจากโฟลเดอร์ Book → output.txt
├── extract_text_from_images.py   # อ่านจากโฟลเดอร์ปัจจุบัน → sum.txt
├── README.md                      # คู่มือนี้
├── Book/                          # โฟลเดอร์เก็บไฟล์ภาพ/text
│   ├── *.png                      # ไฟล์ภาพ
│   ├── *.jpg                      # ไฟล์ภาพ
│   └── *.txt                      # ไฟล์ text
├── output.txt                     # ผลลัพธ์จาก read_book_to_text.py
└── sum.txt                        # ผลลัพธ์จาก extract_text_from_images.py
```

## ✨ คุณสมบัติ

✅ อ่านข้อความจากภาพด้วย OCR ความแม่นยำสูง  
✅ รองรับภาษาไทย + อังกฤษ  
✅ รักษาการจัดย่อหน้าตามต้นฉบับ  
✅ แบ่งหน้าชัดเจน พร้อมหมายเลขหน้า  
✅ รักษาตัวเลข เครื่องหมาย (. - ฯลฯ)  
✅ รองรับไฟล์ภาพ: PNG, JPG, JPEG, BMP, TIFF  
✅ รองรับไฟล์ข้อความ: TXT

## 📊 ตัวอย่างผลลัพธ์

```bash
$ python3 read_book_to_text.py

📚 พบไฟล์ทั้งหมด 8 ไฟล์ในโฟลเดอร์ Book
================================================================================

📄 [1/8] กำลังประมวลผล: Screenshot 2568-12-13 at 16.08.37.png
   🖼️  อ่านจากภาพ...
   ✅ อ่านสำเร็จ: 6,778 ตัวอักษร, 35 บรรทัด

📄 [2/8] กำลังประมวลผล: Screenshot 2568-12-13 at 16.08.42.png
   🖼️  อ่านจากภาพ...
   ✅ อ่านสำเร็จ: 7,252 ตัวอักษร, 45 บรรทัด

...

✅ บันทึกเนื้อหาทั้งหมดลงไฟล์: output.txt
📊 สถิติ:
   - จำนวนไฟล์: 8 ไฟล์
   - จำนวนตัวอักษร: 56,695 ตัวอักษร
   - จำนวนบรรทัด: 336 บรรทัด
```

## 🔍 การแก้ปัญหา

### ไม่พบ tesseract command
```bash
which tesseract  # ตรวจสอบว่าติดตั้งแล้วหรือยัง
brew install tesseract tesseract-lang  # ติดตั้งใหม่
```

### อ่านภาษาไทยไม่ได้
```bash
tesseract --list-langs  # ตรวจสอบภาษาที่รองรับ
brew install tesseract-lang  # ติดตั้งภาษาเพิ่ม
```

### Import Error
```bash
pip3 install --upgrade pillow pytesseract opencv-python numpy
```

## 📝 หมายเหตุ

- ไฟล์ภาพควรมีความคมชัดสูง พื้นหลังสีขาว ตัวอักษรสีดำ จะได้ผลลัพธ์ดีที่สุด
- ไฟล์ output.txt จะถูกเขียนทับทุกครั้งที่รันสคริปต์ใหม่
- การประมวลผลอาจใช้เวลา 5-10 วินาที ต่อภาพ 1 ภาพ

---

**เวอร์ชัน:** 1.0.0 | **วันที่:** December 13, 2025
