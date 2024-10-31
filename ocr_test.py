from pytesseract import pytesseract
import cv2

pytesseract.tesseract_cmd = (
    r"D:\\Tesseract-OCR\\tesseract.exe"
)
img = cv2.imread('1730030896597.jpg')
exp: str = pytesseract.image_to_string(img)
print(exp)