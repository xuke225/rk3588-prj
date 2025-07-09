import time
import cv2
import numpy as np
import onnxruntime
from yoloworld.nms import nms


def read_class_embeddings(embed_path):
    data = np.load(embed_path)
    return data["class_embeddings"], data["class_list"]


class YOLOWorld:

    def __init__(self, path, conf_thres=0.3, iou_thres=0.5, runtime='rknn'):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.runtime = runtime
        print("runtime: ", self.runtime)

        if runtime == 'onnxruntime':
            # Initialize model
            self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            # Get model info
            self.get_input_details()
            self.get_output_details()
        elif runtime == "acl":
            path = path.split(".onnx")[0] + ".om"
            from ais_bench.infer.interface import InferSession
            self.session = InferSession(0, path)
            # Get model info
            self.get_input_details()
            self.get_output_details()
        elif runtime == "rknn":
            from rknn.api import RKNN
            # 加载RKNN模型
            rknn_path = path.replace('.onnx', '.rknn')
            self.rknn = RKNN()
            print(f'Loading RKNN model: {rknn_path}')
            ret = self.rknn.load_rknn(rknn_path)
            if ret != 0:
                print('Load RKNN model failed')
                exit(ret)
            
            # 初始化运行时环境
            print('Init RKNN runtime environment')
            ret = self.rknn.init_runtime(target='rk3588')
            if ret != 0:
                print('Init RKNN runtime environment failed')
                exit(ret)
                
            # 获取模型信息
            self.input_shape = (1, 3, 640, 640)  # 假设输入尺寸
            self.input_height = 640
            self.input_width = 640
            self.num_classes = 80  # 假设类别数量
            
            # # 如果需要确切信息，可以从模型中获取
            # inputs = self.rknn.list_inputs()
            # outputs = self.rknn.list_outputs()
            
            # print(f"RKNN inputs: {inputs}")
            # print(f"RKNN outputs: {outputs}")
            
            # self.input_names = ['images', 'class_embeddings']  # 假设输入名称
            # self.output_names = ['output0', 'output1', 'output2']  # 假设输出名称
        else:
            print("error: runtime must be onnxruntime, acl or rknn")

    def __call__(self, image, class_embeddings):
        return self.detect_objects(image, class_embeddings)

    def detect_objects(self, image, class_embeddings):
        if class_embeddings.shape[1] != self.num_classes:
            print(f"Warning: Number of classes in the class embeddings is {class_embeddings.shape[1]}, model expects {self.num_classes}")

        input_tensor = self.prepare_input(image)

        # Perform yoloworld on the image
        outputs = self.inference(input_tensor, class_embeddings)

        return self.process_output(outputs)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        if self.runtime != 'rknn':
            # Scale input pixel values to 0 to 1
            input_img = input_img.astype(np.float32)/255.0 
        
        if self.runtime == 'rknn':
            # RKNN通常需要NHWC格式
            input_tensor = np.expand_dims(input_img, axis=0)
        else:
            # 其他运行时使用NCHW格式
            input_tensor = cv2.dnn.blobFromImage(input_img)         

        return input_tensor

    def inference(self, input_tensor, class_embeddings):
        start = time.perf_counter()
        if self.runtime == 'onnxruntime':
            outputs = self.session.run(self.output_names,
                                      {self.input_names[0]: input_tensor, self.input_names[1]: class_embeddings})
        elif self.runtime == "acl":
            outputs = self.session.infer([input_tensor, class_embeddings])
        elif self.runtime == "rknn":
            # RKNN推理
            outputs = self.rknn.inference(inputs=[input_tensor, class_embeddings])
        else:
            print("error: runtime must be onnxruntime, acl or rknn")
            return None
            
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        
        for out in outputs:
            print(out.shape)
        print(outputs[0][0,0,:10])
        print(outputs[1][0,0,0,:10])
        print(outputs[2][0,0,0,:10])
        
        return outputs
        # for i, out in enumerate(outputs):
        #     print(f"Output {i} shape: {out.shape}")
        
        # if outputs[0].size > 10:
        #     print(f"Output 0 first 10 values: {outputs[0].flatten()[:10]}")
        # if len(outputs) > 1 and outputs[1].size > 10:
        #     print(f"Output 1 first 10 values: {outputs[1].flatten()[:10]}")
        # if len(outputs) > 2 and outputs[2].size > 10:
        #     print(f"Output 2 first 10 values: {outputs[2].flatten()[:10]}")
        
        # return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = YOLOWorld.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.num_classes = model_inputs[1].shape[1]

        print("input_shape: ", self.input_shape)
        print("num_classes: ", self.num_classes)
        print("input_names: ", self.input_names)

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        print("output_names: ", self.output_names)
    
    # def __del__(self):
    #     if self.runtime == "rknn":
    #         if hasattr(self, 'rknn'):
    #             self.rknn.release()


if __name__ == '__main__':
    from yoloworld.DetectionDrawer import DetectionDrawer
    try:
        from imread_from_url import imread_from_url
    except ImportError:
        def imread_from_url(url):
            print(f"Cannot download image from {url}, using local image instead")
            return cv2.imread("../data/panda.jpg")

    # 使用RKNN模型
    model_path = "../models/yolov8s-worldv2.rknn"
    embed_path = "../data/panda_embeddings.npz"

    # Load class embeddings
    class_embeddings, class_list = read_class_embeddings(embed_path)

    # Initialize YOLO-World object detector with RKNN runtime
    yoloworld_detector = YOLOWorld(model_path, conf_thres=0.3, iou_thres=0.5, runtime='rknn')

    # Initialize DetectionDrawer
    drawer = DetectionDrawer(class_list)

    # 尝试从URL加载，如果失败则使用本地图像
    try:
        img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
        img = imread_from_url(img_url)
    except:
        img = cv2.imread("../test_data/test.jpg")
        if img is None:
            print("Could not load image, please check the path")
            exit(1)

    # Detect Objects
    boxes, scores, class_ids = yoloworld_detector(img, class_embeddings)

    # Draw detections
    combined_img = drawer(img, boxes, scores, class_ids)
    
    # 保存输出图像
    cv2.imwrite("yoloworld_output.jpg", combined_img)
    print("Detection results saved to yoloworld_output.jpg")
    
    # 如果有显示环境可以显示图像
    try:
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", combined_img)
        cv2.waitKey(0)
    except:
        print("Cannot display image (no display environment)")
