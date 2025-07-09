import numpy as np
import cv2
from rknn.api import RKNN
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
 

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)s

CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class Model:
    """object detection model class for handling inference and visualization."""

    def __init__(self, input_image):
        """
         Args:
            modelpath: Path to the om model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.input_image = input_image
        self.confidence_thres = OBJ_THRESH
        self.iou_thres = NMS_THRESH
        self.input_width = IMG_SIZE[0]
        self.input_height = IMG_SIZE[1]

        # Load the class names from the COCO dataset
        self.classes =  CLASSES

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Get the height and width of the input image
        self.img_height, self.img_width = self.input_image.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        return img
 
    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        detects = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            x1, y1, w, h = boxes[i]
            detect = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x1 + w),
                "y2": int(y1 + h),
                "class_score": float(scores[i]),
                "class_name": self.classes[class_ids[i]] 
            }
            detects.append(detect)

        # Return the modified input image
        return detects


def load_rknn(rknn_path: str, core_id: int = 0, target='rk3588'):
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    assert ret == 0

    if core_id == 0:
        ret = rknn.init_runtime(target=target,core_mask=RKNN.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn.init_runtime(target=target,core_mask=RKNN.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn.init_runtime(target=target,core_mask=RKNN.NPU_CORE_2)
    elif core_id == -1:
        ret = rknn.init_runtime(target=target,core_mask=RKNN.NPU_CORE_0_1_2)
    else:
        ret = rknn.init_runtime(target=target)
    assert ret == 0
    return rknn

class rknnPoolExecutor:
    def __init__(self, rknn_path, num_thread, func):
        assert num_thread > 0
        self.num_thread = num_thread
        self.queue = Queue()
        self.rknnPool = [load_rknn(rknn_path, i % 3) for i in range(num_thread)]
        self.pool = ThreadPoolExecutor(max_workers=num_thread)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(
            self.pool.submit(
                self.func, self.rknnPool[self.num % self.num_thread], frame
            ))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        # print(fut.result())
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn in self.rknnPool:
            rknn.release()

def infer(rknn:RKNN, image:np.ndarray):     
    model = Model(image)
    img_data = model.preprocess()
    outputs = rknn.inference(inputs=[img_data])
    detects = model.postprocess(outputs)        
    # print(detects)
    return detects
