import cv2
import os

def initialize_camera(frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None

    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, 150)  
    return cap

def detect_plate(image): 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    

    edged = cv2.Canny(bilateral, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    roi = None
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h

            if 2.0 <= aspect_ratio <= 5.5:
                plate_contour = approx
                roi = image[y:y+h, x:x+w]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "License Plate", (x, y - 5),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                break
    
    return roi, plate_contour

def enhance_plate(plate_img):
    if plate_img is None:
        return None

    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    return denoised

def main():
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    if not os.path.exists("Plates"):
        os.makedirs("Plates")
    
    
    cap = initialize_camera(FRAME_WIDTH, FRAME_HEIGHT)
    if cap is None:
        return
    
    print("Improved License Plate Detection Started!")
    print("Press 's' to save detected plate")
    print("Press 'q' to quit")
    
    count = 0
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("Error: Failed to grab frame")
            break
            
        try:

            frame_copy = img.copy()
            plate_roi, plate_contour = detect_plate(frame_copy)
            
            cv2.imshow("Result", frame_copy)
            
            
            if plate_roi is not None:
                enhanced_plate = enhance_plate(plate_roi)
                
                if enhanced_plate is not None:
                    cv2.imshow("Detected Plate", enhanced_plate)    
                
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('s'):
                        cv2.imwrite(f"Plates/plate_original_{count}.jpg", plate_roi)
                        cv2.imwrite(f"Plates/plate_enhanced_{count}.jpg", enhanced_plate)
                        print(f"Saved plate images #{count}")
                        count += 1
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()
main()