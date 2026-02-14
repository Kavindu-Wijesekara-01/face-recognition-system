import cv2
import math
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        print("Loading YOLO Model...")
        self.model = YOLO("yolov8n.pt")
        
        # වේගය වැඩි කරන රහස: මතකය (Cache)
        self.last_results = []  # අන්තිමට හොයාගත්ත කොටු ටික මෙතන තියාගන්නවා
        self.frame_count = 0    # Frame ගණන් කරන්න

    def detect_and_draw(self, frame):
        self.frame_count += 1

        # --- AI RUN LOGIC (SKIP FRAMES) ---
        # හැම Frame 5 කට වරක් විතරක් AI එක රන් කරනවා.
        # (ඔයාගේ මැෂින් එක තව Slow නම් 5 වෙනුවට 10 දාන්න)
        if self.frame_count % 5 == 0:
            
            # පින්තූරය පොඩි කරලා යවනවා (වේගය 2x වැඩි වෙයි)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            results = self.model(small_frame, stream=True, verbose=False)
            
            # පරණ මෙමරි එක සුද්ද කරනවා
            self.last_results = [] 

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # කුඩා පින්තූරයේ ඛණ්ඩාංක (Coordinates)
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # අපි යැව්වේ භාගයක් (0.5) පොඩි කරපු එකක් නිසා,
                    # එන උත්තරේ ආපහු 2න් වැඩි කරන්න ඕන (Original Size එකට)
                    x1 = int(x1 * 2)
                    y1 = int(y1 * 2)
                    x2 = int(x2 * 2)
                    y2 = int(y2 * 2)

                    # නම සහ විශ්වාසය
                    cls_id = int(box.cls[0])
                    current_class = self.model.names[cls_id]
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # --- නම වෙනස් කරන තැන ---
                    if current_class == 'cell phone': current_class = 'Phone'
                    elif current_class == 'remote': current_class = 'Remote'
                    elif current_class == 'bottle': current_class = 'Bottle'

                    # පාට
                    color = (255, 0, 255)

                    # මෙමරි එකට දාගන්නවා (ඊළඟ Frame 4 දී මේකම අඳින්න)
                    self.last_results.append((x1, y1, x2, y2, current_class, conf, color))

        # --- DRAWING LOGIC (ALWAYS) ---
        # AI රන් වුනත් නැතත්, මෙමරි එකේ තියෙන කොටු ටික අඳිනවා
        for (x1, y1, x2, y2, label, conf, color) in self.last_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf}%', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame