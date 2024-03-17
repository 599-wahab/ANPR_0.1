import cv2
import easyocr

harcascade = "model/haarcascade_russian_plate_number.xml"

# Change 'video_file.mp4' to the path of your video file
cap = cv2.VideoCapture('test video german.mp4')

cap.set(3, 640) # width
cap.set(4, 480) # height

min_area = 500
count = 0

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

            # Perform OCR on the number plate region using EasyOCR
            results = reader.readtext(img_roi)
            plate_text = results[0][1] if results else "No text detected"
            print("Detected Plate Text:", plate_text)
    
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1
