import os
import time
import uuid
import cv2
import threading
import numpy as np
import Levenshtein

from dotenv import dotenv_values
from easyocr import Reader
from datetime import datetime
from sort import Sort
from ultralytics import YOLO


OCR_MIN_PROB = 0.7
MODEL_MIN_PROB = 0.25
COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
]

class Server(object):

    image_queue = []
    track_dict = {}
    running = False

    def __init__(self):
        # Init model
        self.model = YOLO("yolov8n-v4.pt")

        # Init ocr
        self.reader = Reader(["fr"])
        self.bib_traker = Sort()
        
        self.config = dotenv_values(".env")
        self.bibs_list = open('bibs.txt', 'r').read().split('\n')

    def cleanup_text(self, text):
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def process_bib(self, track_id, cropped_image):
        if track_id not in self.track_dict:
            self.track_dict[track_id] = {}

        # Use the OCR reader to read text from the cropped image
        results = self.reader.readtext(cropped_image, canvas_size=200, min_size=30 ,text_threshold=OCR_MIN_PROB, link_threshold=0.3, low_text=0.3)

        # Loop over the OCR results
        for (bbox, text, prob) in results:
            # Clean up the text
            text = self.cleanup_text(text)

            if text.isdigit():
                # Find closest values in bib_list
                min_distance = min([Levenshtein.distance(text, x) for x in self.bibs_list])
                closests = [x for x in self.bibs_list if Levenshtein.distance(text, x) <= min_distance]

                # Add match in track_dict
                for val in closests:
                    if val not in self.track_dict[track_id]:
                        self.track_dict[track_id][val] = 0
                    self.track_dict[track_id][val] += prob

        # If self.track_dict[track_id] is empty, return None
        if not self.track_dict[track_id]:
            return ''
            
        # Return the bib with highest score
        return max(self.track_dict[track_id], key=self.track_dict[track_id].get)

    def process_image(self, img):
        start_time = time.time()
        
        # Inference
        results = self.model(img, conf=MODEL_MIN_PROB)

        # Update traker
        track_bbs_ids = self.bib_traker.update(results[0].boxes.data.cpu().numpy())   

        for track_bb_id in track_bbs_ids:
            xA, yA, xB, yB, track_id = track_bb_id.astype(int)

            # Crop the original image to get the region of interest
            cropped_image = img[yA:yB, xA:xB]

            bib = self.process_bib(track_id, cropped_image)

            # draw rectangle on the image
            cv2.rectangle(img, (xA, yA), (xB, yB), COLORS[track_id%6], 3)
            
            # draw text on the image 
            text = f'id {track_id} : {bib}'
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=3)
            cv2.rectangle(img, (xA, yA - text_size[1]), (xA + text_size[0], yA), COLORS[track_id%6], -1)
            cv2.putText(img, text, (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            
            
        end_time = time.time()
        cv2.putText(img, f'{int(1/(end_time - start_time))} FPS', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('result', cv2.resize(img, (1152,648)))
        cv2.waitKey(1)

    def worker(self): 
        while self.running:
            self.lock.acquire()
            if len(self.image_queue) > 0:
                item = self.image_queue.pop(0)
                self.lock.release()
                image = cv2.imread(item['path'])
                os.remove(item['path'])
                self.process_image(image)
            else:
                self.lock.release()

    def run(self):
        self.running = True

        # Init worker thread
        self.lock = threading.RLock()
        thread = threading.Thread(target=self.worker)
        # thread.start()
        
        # Init camera
        cam = cv2.VideoCapture(self.config['SOURCE'])

        if cam.isOpened() == False:
            print("Error")

        ret = True
        while ret:
            ret, img = cam.read()

            self.process_image(img)
            
            # self.lock.acquire()
            # item = {
            #     'path': f'/tmp/{uuid.uuid4().hex}.png',
            #     'date': datetime.now()
            # } 
            # cv2.imwrite(item['path'], img)
            # self.image_queue.append(item)
            # self.lock.release()
            

if __name__ == "__main__":
    reader = Server()
    reader.run()