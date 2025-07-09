import cv2
from yoloworld import YOLOWorld, DetectionDrawer, read_class_embeddings

model_path = "models/yolov8s-worldv2.rknn"
embed_path = "class_data/panda_embeddings.npz"

# Load class embeddings
class_embeddings, class_list = read_class_embeddings(embed_path)
print("Detecting classes:", class_list)

# Initialize YOLO-World object detector
yoloworld_detector = YOLOWorld(model_path, conf_thres=0.005, iou_thres=0.5, runtime='rknn')

# Initialize DetectionDrawer
drawer = DetectionDrawer(class_list)

# img = cv2.imread("./test_data/test.jpg")
img = cv2.imread("/home/orangepi/EagleCV/Algorithms/yolo_world/test_data/test.jpg")

# Detect Objects
boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

# Draw detections
combined_img = drawer(img, boxes, scores, class_ids)
# cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
# cv2.imshow("Output", combined_img)
# cv2.waitKey(0)

cv2.imwrite("/home/orangepi/EagleCV/Algorithms/yolo_world/test_data/result.jpg", combined_img)
