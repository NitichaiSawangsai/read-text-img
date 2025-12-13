#!/usr/bin/env python3
"""
สคริปต์สำหรับอ่านข้อความจากภาพทั้งหมดในโฟลเดอร์
ใช้ pytesseract และ OpenCV สำหรับการประมวลผลภาพที่แม่นยำ
รักษาการจัดรูปแบบ ย่อหน้า และความถูกต้อง 100%
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
    subprocess.check_call(["pip", "install", "pillow", "pytesseract", "opencv-python", "numpy"])
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np

def preprocess_image_high_quality(image_path):
    """
    ประมวลผลภาพเพื่อเพิ่มความแม่นยำสูงสุดในการอ่าน OCR
    รักษารายละเอียดตัวอักษร ตัวเลข และเครื่องหมายวรรคตอน
    """
    # อ่านภาพด้วย OpenCV
    img = cv2.imread(image_path)
    
    # เพิ่มขนาดภาพ 2 เท่า สำหรับความละเอียดที่ดีขึ้น
    height, width = img.shape[:2]
    img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ลด noise เล็กน้อยแต่รักษารายละเอียด
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # ปรับ contrast อย่างอ่อน
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # ใช้ Otsu's thresholding สำหรับแยกข้อความอัตโนมัติ
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def preprocess_image(image_path):
    """
    ประมวลผลภาพเพื่อเพิ่มความแม่นยำในการอ่าน OCR
    """
    # อ่านภาพด้วย OpenCV
    img = cv2.imread(image_path)
    
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # เพิ่ม contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # ลด noise ด้วย bilateral filter
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Adaptive thresholding เพื่อแยกข้อความ
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # เพิ่มความคมชัด
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened

def extract_text_from_image(image_path):
    """
    อ่านข้อความจากภาพด้วย pytesseract โดยใช้การตั้งค่าที่แม่นยำ
    รองรับทั้งภาษาไทยและภาษาอังกฤษ
    """
    try:
        # ประมวลผลภาพก่อน
        processed_img = preprocess_image(image_path)
        
        # ลองหลายวิธีเพื่อความแม่นยำสูงสุด
        
        # วิธีที่ 1: ภาษาไทย + อังกฤษ ด้วย config พิเศษ
        custom_config = r'--oem 3 --psm 6 -l tha+eng'
        text1 = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # วิธีที่ 2: ใช้ภาพต้นฉบับ
        img_pil = Image.open(image_path)
        text2 = pytesseract.image_to_string(img_pil, lang='tha+eng', config='--oem 3 --psm 6')
        
        # วิธีที่ 3: ภาษาอังกฤษอย่างเดียว (สำหรับกรณีที่เป็นภาษาอังกฤษล้วน)
        text3 = pytesseract.image_to_string(img_pil, lang='eng', config='--oem 3 --psm 6')
        
        # เลือกผลลัพธ์ที่ได้ข้อความมากที่สุด
        texts = [text1, text2, text3]
        best_text = max(texts, key=len)
        
        return best_text.strip()
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการอ่านภาพ {image_path}: {e}")
        return ""

def main():
    # กำหนดโฟลเดอร์ปัจจุบัน
    current_dir = Path(__file__).parent
    
    # ค้นหาไฟล์ภาพทั้งหมด
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(str(current_dir / ext)))
    
    # เรียงลำดับไฟล์
    image_files.sort()
    
    if not image_files:
        print("ไม่พบไฟล์ภาพในโฟลเดอร์")
        return
    
    print(f"พบภาพทั้งหมด {len(image_files)} ไฟล์")
    print("=" * 80)
    
    # เก็บข้อความทั้งหมด
    all_text = []
    
    # อ่านข้อความจากแต่ละภาพ
    for idx, image_path in enumerate(image_files, 1):
        filename = Path(image_path).name
        print(f"\n[{idx}/{len(image_files)}] กำลังประมวลผล: {filename}")
        
        text = extract_text_from_image(image_path)
        
        if text:
            print(f"✓ อ่านได้ {len(text)} ตัวอักษร")
            all_text.append(f"{'=' * 80}")
            all_text.append(f"ไฟล์: {filename}")
            all_text.append(f"{'=' * 80}")
            all_text.append(text)
            all_text.append("")  # บรรทัดว่าง
        else:
            print(f"✗ ไม่พบข้อความในภาพนี้")
    
    # บันทึกลงไฟล์ sum.txt
    output_file = current_dir / "sum.txt"
    
    if all_text:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        print("\n" + "=" * 80)
        print(f"✓ บันทึกข้อความทั้งหมดลงไฟล์: {output_file}")
        print(f"✓ จำนวนบรรทัดทั้งหมด: {len(all_text)}")
        print("=" * 80)
    else:
        print("\n✗ ไม่มีข้อความที่อ่านได้จากภาพ")

if __name__ == "__main__":
    main()
