#!/usr/bin/env python3
"""
สคริปต์สำหรับอ่านภาพและไฟล์ text จากโฟลเดอร์ Book
รักษาความแม่นยำ 100% รวมถึงตัวเลข เครื่องหมาย และการจัดย่อหน้า
บันทึกผลลัพธ์ลง raw-output.txt
"""

import os
import glob
import re
from pathlib import Path
try:
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
except ImportError:
    print("กำลังติดตั้ง libraries ที่จำเป็น...")
    import subprocess
    subprocess.check_call(["pip3", "install", "pillow", "pytesseract", "opencv-python", "numpy"])
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np


def preprocess_for_high_accuracy(image_path):
    """
    ประมวลผลภาพเพื่อความแม่นยำสูงสุด
    เน้นการรักษาตัวเลข เครื่องหมาย และรูปแบบต้นฉบับ
    """
    # อ่านภาพ
    img = cv2.imread(image_path)
    
    # ขยายภาพ 2 เท่าเพื่อเพิ่มความละเอียด
    height, width = img.shape[:2]
    img_scaled = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    
    # ลด noise เล็กน้อย
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # ปรับ contrast อย่างนุ่มนวล
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # ใช้ Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def extract_text_with_layout(image_path):
    """
    อ่านข้อความจากภาพพร้อมรักษาโครงสร้างและย่อหน้า
    รองรับภาษาไทยและอังกฤษ รักษาตัวเลขและเครื่องหมายวรรคตอน
    """
    try:
        # ประมวลผลภาพ
        processed = preprocess_for_high_accuracy(image_path)
        
        # อ่านภาพต้นฉบับ
        img_original = Image.open(image_path)
        
        # Config สำหรับ OCR ที่แม่นยำสูง
        # PSM 6 = Uniform block of text (เหมาะกับเอกสารทั่วไป)
        # PSM 3 = Fully automatic page segmentation (รักษาโครงสร้าง)
        
        configs = [
            # Config 1: ภาษาไทย+อังกฤษ รักษาโครงสร้าง
            ('tha+eng', '--oem 3 --psm 6'),
            # Config 2: ภาษาไทย+อังกฤษ จากภาพที่ประมวลผล
            ('tha+eng', '--oem 3 --psm 3'),
            # Config 3: ภาษาอังกฤษ (สำหรับเอกสารที่เป็นภาษาอังกฤษล้วน)
            ('eng', '--oem 3 --psm 6'),
        ]
        
        results = []
        
        # ลอง config 1: ภาพต้นฉบับ
        text1 = pytesseract.image_to_string(img_original, lang='tha+eng', config='--oem 3 --psm 6')
        results.append(text1)
        
        # ลอง config 2: ภาพที่ประมวลผลแล้ว
        text2 = pytesseract.image_to_string(processed, lang='tha+eng', config='--oem 3 --psm 6')
        results.append(text2)
        
        # ลอง config 3: ภาพต้นฉบับ + PSM 3 (รักษาโครงสร้างหน้า)
        text3 = pytesseract.image_to_string(img_original, lang='tha+eng', config='--oem 3 --psm 3')
        results.append(text3)
        
        # ลอง config 4: อังกฤษอย่างเดียว (กรณีภาษาอังกฤษล้วน)
        text4 = pytesseract.image_to_string(img_original, lang='eng', config='--oem 3 --psm 6')
        results.append(text4)
        
        # เลือกผลลัพธ์ที่ดีที่สุด (มีเนื้อหามากที่สุดและมีโครงสร้างที่ดี)
        best_text = max(results, key=lambda x: len(x.strip()))
        
        # ทำความสะอาดข้อความเล็กน้อย แต่รักษาย่อหน้าและเครื่องหมาย
        text = best_text.strip()
        
        # รักษาบรรทัดว่างที่มีมากกว่า 2 บรรทัดให้เหลือ 2 บรรทัด
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอ่านภาพ {image_path}: {e}")
        return ""


def read_text_file(file_path):
    """
    อ่านไฟล์ text ธรรมดา
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # ลองเข้ารหัสอื่น
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            print(f"❌ ไม่สามารถอ่านไฟล์ {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ {file_path}: {e}")
        return ""


def format_page_separator(page_num, filename):
    """
    สร้างตัวแบ่งหน้าที่สวยงาม
    """
    separator = "=" * 100
    header = f"หน้าที่ {page_num}: {filename}"
    return f"\n\n{separator}\n{header}\n{separator}\n\n"


def main():
    # กำหนดโฟลเดอร์
    script_dir = Path(__file__).parent
    book_folder = script_dir / "Book"
    output_file = script_dir / "raw-output.txt"
    
    if not book_folder.exists():
        print(f"❌ ไม่พบโฟลเดอร์ Book ที่: {book_folder}")
        return
    
    # ค้นหาไฟล์ทั้งหมดในโฟลเดอร์ Book
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp', '*.tiff']
    text_extensions = ['*.txt', '*.TXT']
    
    all_files = []
    
    # รวมไฟล์ภาพ
    for ext in image_extensions:
        all_files.extend(list(book_folder.glob(ext)))
    
    # รวมไฟล์ text
    for ext in text_extensions:
        all_files.extend(list(book_folder.glob(ext)))
    
    # เรียงลำดับตามชื่อไฟล์
    all_files.sort(key=lambda x: x.name)
    
    if not all_files:
        print(f"❌ ไม่พบไฟล์ใดๆ ในโฟลเดอร์ Book")
        return
    
    print(f"📚 พบไฟล์ทั้งหมด {len(all_files)} ไฟล์ในโฟลเดอร์ Book")
    print("=" * 100)
    
    # เก็บเนื้อหาทั้งหมด
    all_content = []
    
    # ประมวลผลแต่ละไฟล์
    for idx, file_path in enumerate(all_files, 1):
        filename = file_path.name
        file_extension = file_path.suffix.lower()
        
        print(f"\n📄 [{idx}/{len(all_files)}] กำลังประมวลผล: {filename}")
        
        content = ""
        
        # ตรวจสอบประเภทไฟล์
        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            # อ่านจากภาพ
            print(f"   🖼️  อ่านจากภาพ...")
            content = extract_text_with_layout(str(file_path))
        elif file_extension in ['.txt']:
            # อ่านจากไฟล์ text
            print(f"   📝 อ่านจากไฟล์ text...")
            content = read_text_file(str(file_path))
        
        if content:
            char_count = len(content)
            line_count = content.count('\n') + 1
            print(f"   ✅ อ่านสำเร็จ: {char_count} ตัวอักษร, {line_count} บรรทัด")
            
            # เพิ่มตัวแบ่งหน้า
            page_separator = format_page_separator(idx, filename)
            all_content.append(page_separator)
            all_content.append(content)
        else:
            print(f"   ⚠️  ไม่พบเนื้อหาในไฟล์นี้")
    
    # บันทึกลงไฟล์ raw-output.txt
    if all_content:
        final_text = ''.join(all_content)
        
        # เพิ่มหัวเรื่องและวันที่
        header = f"""{'=' * 100}
📚 สรุปเนื้อหาจากโฟลเดอร์ Book
{'=' * 100}
จำนวนไฟล์ทั้งหมด: {len(all_files)} ไฟล์
วันที่ประมวลผล: {Path(output_file).stat().st_mtime if output_file.exists() else 'N/A'}
{'=' * 100}
"""
        
        final_output = header + final_text
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_output)
        
        total_chars = len(final_text)
        total_lines = final_text.count('\n') + 1
        
        print("\n" + "=" * 100)
        print(f"✅ บันทึกเนื้อหาทั้งหมดลงไฟล์: {output_file}")
        print(f"📊 สถิติ:")
        print(f"   - จำนวนไฟล์: {len(all_files)} ไฟล์")
        print(f"   - จำนวนตัวอักษร: {total_chars:,} ตัวอักษร")
        print(f"   - จำนวนบรรทัด: {total_lines:,} บรรทัด")
        print("=" * 100)
        print("\n✨ เสร็จสมบูรณ์! สามารถเปิดไฟล์ raw-output.txt เพื่อดูผลลัพธ์ได้เลยครับ")
    else:
        print("\n❌ ไม่มีเนื้อหาที่อ่านได้จากไฟล์ใดๆ")


if __name__ == "__main__":
    main()
