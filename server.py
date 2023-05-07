import time
import uuid
import cv2
import queue
import threading
import numpy as np
import Levenshtein
import requests
import json
import calendar
import base64

from dotenv import dotenv_values
from easyocr import Reader
from datetime import datetime
from datetime import timedelta
from sort import Sort
from ultralytics import YOLO

QUEUE_MAX_SIZE = 100
OCR_MIN_PROB = 0.7
MODEL_MIN_PROB = 0.10
COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]

class Server(object):

    def __init__(self):
        self.config = dotenv_values(".env")
        self.bibs_list = open('bibs.txt', 'r').read().split('\n')

        # creation de la course sur API
        try:
            current_GMT = time.gmtime()
            time_stamp = calendar.timegm(current_GMT)
            r = requests.post("http://" + self.config['ADRESSE_API'] + ":" + self.config['PORT_API'] + "/course", data={"nom": self.config['NOM_COURSE'], 'timestamp': time_stamp})
            self.id_course = json.loads(r.text)['id']
        except:
            print("Error : " + r.text)
            exit()

        # Init model
        self.model = YOLO(self.config["MODEL"])

        # Init ocr
        self.reader = Reader(["fr"])
        self.bib_traker = Sort()

        self.queue = queue.Queue(QUEUE_MAX_SIZE)
        self.lock = threading.Lock()
        self.image_queue = []
        self.track_dict = {}

    def cleanup_text(self, text):
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def process_bib(self, track_id, cropped_image, img, date):
        with self.lock:
            if track_id not in self.track_dict:
                self.track_dict[track_id] = {
                    'matches': {},
                    'first_image': np.copy(img),
                    'creation_date': date,
                }
        
        self.track_dict[track_id]['last_detection'] = datetime.now()
        self.track_dict[track_id]['last_image'] = np.copy(img)

        # Use the OCR reader to read text from the cropped image
        results = self.reader.readtext(cropped_image, canvas_size=200, min_size=30 ,text_threshold=OCR_MIN_PROB, link_threshold=0.3, low_text=0.3)

        # Loop over the OCR results
        for (bbox, text, prob) in results:
            # Clean up the text
            text = self.cleanup_text(text)

            if text.isdigit():
                # Find closest values in bibs_list
                min_distance = min([Levenshtein.distance(text, x) for x in self.bibs_list])
                matches = [x for x in self.bibs_list if Levenshtein.distance(text, x) <= min_distance]

                if len(matches) > 5:
                    # If too many matches then ignore result
                    continue

                # Add matches in track_dict
                for val in matches:
                    if val not in self.track_dict[track_id]['matches']:
                        self.track_dict[track_id]['matches'][val] = 0
                    self.track_dict[track_id]['matches'][val] += prob
        
        # Return the bib with highest score
        return max(self.track_dict[track_id]['matches'], default='', key=self.track_dict[track_id]['matches'].get)

    def process_image(self, img, date):
        start_time = time.time()
        
        
        # Inference
        results = self.model(img, conf=MODEL_MIN_PROB)

        # Update traker
        track_bbs_ids = self.bib_traker.update(results[0].boxes.data.cpu().numpy())   

        for track_bb_id in track_bbs_ids:
            xA, yA, xB, yB, track_id = track_bb_id.astype(int)

            # Crop the original image to get the region of interest
            cropped_image = img[yA:yB, xA:xB]

            bib = self.process_bib(track_id, cropped_image, img, date)
            
            # draw rectangle on the image
            cv2.rectangle(img, (xA, yA), (xB, yB), COLORS[track_id%5], 3)
            
            # draw text on the image 
            text = f'id {track_id} : {bib}'
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=3)
            cv2.rectangle(img, (xA, yA - text_size[1]), (xA + text_size[0], yA), COLORS[track_id%5], -1)
            cv2.putText(img, text, (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
            
            
        end_time = time.time()
        cv2.putText(img, f'{int(1/(end_time - start_time))} FPS', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('result', cv2.resize(img, (1152,648)))
        cv2.waitKey(1)

    def process_images(self):
        while True:
            item = self.queue.get()
            self.process_image(item['image'], item['date'])
            self.queue.task_done()

    def read_video(self):
        # Init camera
        cam = cv2.VideoCapture(self.config['SOURCE'])

        if cam.isOpened() == False:
            print("Error")

        while True:
            ret, img = cam.read()

            if not ret:
                break

            self.queue.put({
                #'path': f'/tmp/{uuid.uuid4().hex}.png',
                'image': img,
                'date': datetime.now()
            })

    def send_to_api(self):
        bibs_sent = []
        while True:
            time.sleep(10)
            with self.lock:
                for track_id in list(self.track_dict):
                    # if last detection is 10 sec old
                    if self.track_dict[track_id]['last_detection'] < (datetime.now() - timedelta(seconds=10)):
                        bib_number = max(self.track_dict[track_id]['matches'], default='', key=self.track_dict[track_id]['matches'].get)

                        if bib_number != '' and bib_number not in bibs_sent:
                            # send to api
                            bibs_sent.append(bib_number)

                            # encode the image
                            _, buffer = cv2.imencode('.jpg', self.track_dict[track_id]['last_image'])
                            base64_image = base64.b64encode(buffer).decode('utf-8')
                            creation_date = int(self.track_dict[track_id]['creation_date'].timestamp())

                            # Requête à l'API pour enregistrer le passage du coureur
                            requests.post("http://" + self.config['ADRESSE_API'] + ":" + self.config['PORT_API'] + "/pointControle", data={"dossard": bib_number, "timestamp": creation_date, "distance": self.config['KILOMETRE'], "courseId": self.id_course, "image": base64_image}) 
                        del self.track_dict[track_id]

    def start(self):
        t1 = threading.Thread(target=self.read_video)
        t2 = threading.Thread(target=self.process_images)
        t3 = threading.Thread(target=self.send_to_api)

        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
            

if __name__ == "__main__":
    reader = Server()
    reader.start()