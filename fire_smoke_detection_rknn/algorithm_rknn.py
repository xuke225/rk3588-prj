import numpy as np
import cv2
from rknn.api import RKNN
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import os

# 火灾烟雾检测参数配置
OBJ_THRESH = 0.5  # 置信度阈值
NMS_THRESH = 0.4  # NMS阈值
IMG_SIZE = (640, 640)  # 输入图像尺寸 (width, height)

# 火灾烟雾类别映射
FIRE_SMOKE_CLASSES = {
    0: 'Fire',   # 火灾
    1: 'smoke'   # 烟雾
}

# 类别中文名映射
FIRE_SMOKE_CLASSES_CN = {
    0: '火灾',
    1: '烟雾'
}

class FireSmokeModel:
    """火灾烟雾检测模型类，用于处理推理和结果解析"""

    def __init__(self, input_image):
        """
        Args:
            input_image: 输入图像
        """
        self.input_image = input_image
        self.confidence_thres = OBJ_THRESH
        self.iou_thres = NMS_THRESH
        self.input_width = IMG_SIZE[0]
        self.input_height = IMG_SIZE[1]

        # 加载火灾烟雾类别名称
        self.classes = FIRE_SMOKE_CLASSES
        self.classes_cn = FIRE_SMOKE_CLASSES_CN

    def preprocess(self):
        """
        图像预处理

        Returns:
            image_data: 预处理后的图像数据，准备进行推理
        """
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.input_image.shape[:2]

        # 将图像颜色空间从BGR转换为RGB
        img = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配输入形状
        img = cv2.resize(img, (self.input_width, self.input_height))
        
        return img
 
    def postprocess(self, output):
        """
        对模型输出进行后处理，提取边界框、分数和类别ID

        Args:
            output (numpy.ndarray): 模型的输出

        Returns:
            list: 检测结果列表
        """
        # 转置并压缩输出以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取输出数组中的行数
        rows = outputs.shape[0]

        # 存储检测的边界框、分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组中的每一行
        for i in range(rows):
            # 从当前行提取类别分数
            classes_scores = outputs[i][4:]

            # 找到类别分数中的最大分数
            max_score = np.amax(classes_scores)

            # 如果最大分数高于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取具有最高分数的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 将类别ID、分数和边界框坐标添加到相应的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非最大抑制来过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 遍历非最大抑制后选择的索引
        detects = []
        for i in indices:
            # 获取对应索引的边界框、分数和类别ID
            x1, y1, w, h = boxes[i]
            class_name = self.classes.get(class_ids[i], f"class_{class_ids[i]}")
            class_name_cn = self.classes_cn.get(class_ids[i], class_name)
            
            detect = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x1 + w),
                "y2": int(y1 + h),
                "class_score": float(scores[i]),
                "class_name": class_name,
                "class_name_cn": class_name_cn,
                "class_id": int(class_ids[i])
            }
            detects.append(detect)

        return detects


def load_rknn(rknn_path: str, core_id: int = 0, target='rk3588'):
    """
    加载RKNN模型

    Args:
        rknn_path: RKNN模型文件路径
        core_id: NPU核心ID (0, 1, 2, -1)
        target: 目标平台

    Returns:
        RKNN: 初始化后的RKNN实例
    """
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    assert ret == 0

    if core_id == 0:
        ret = rknn.init_runtime(target=target, core_mask=RKNN.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn.init_runtime(target=target, core_mask=RKNN.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn.init_runtime(target=target, core_mask=RKNN.NPU_CORE_2)
    elif core_id == -1:
        ret = rknn.init_runtime(target=target, core_mask=RKNN.NPU_CORE_0_1_2)
    else:
        ret = rknn.init_runtime(target=target)
    assert ret == 0
    return rknn


class FireSmokeRknnPoolExecutor:
    """RKNN线程池执行器，用于并发处理火灾烟雾检测"""
    
    def __init__(self, rknn_path, num_thread, func):
        assert num_thread > 0
        self.num_thread = num_thread
        self.queue = Queue()
        self.rknnPool = [load_rknn(rknn_path, i % 3) for i in range(num_thread)]
        self.pool = ThreadPoolExecutor(max_workers=num_thread)
        self.func = func
        self.num = 0

    def put(self, frame):
        """提交推理任务"""
        self.queue.put(
            self.pool.submit(
                self.func, self.rknnPool[self.num % self.num_thread], frame
            ))
        self.num += 1

    def get(self):
        """获取推理结果"""
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        """释放资源"""
        self.pool.shutdown()
        for rknn in self.rknnPool:
            rknn.release()


def fire_smoke_infer(rknn: RKNN, image: np.ndarray):
    """
    火灾烟雾检测推理函数

    Args:
        rknn: RKNN实例
        image: 输入图像

    Returns:
        list: 检测结果列表
    """
    model = FireSmokeModel(image)
    img_data = model.preprocess()
    outputs = rknn.inference(inputs=[img_data])
    detects = model.postprocess(outputs)
    return detects


class FireSmokeDetectorRKNN:
    """
    基于RKNN的火灾烟雾检测算法
    专为瑞芯微芯片(如RK3588)优化的火灾和烟雾实时检测
    """
    
    def __init__(self, model_path: str = None, **kwargs):
        """
        初始化RKNN火灾烟雾检测算法
        Args:
            model_path: RKNN模型文件路径
            **kwargs: 其他配置参数
        """
        self.model_path = model_path or "model/yolo_fire.rknn"
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.nms_threshold = kwargs.get('nms_threshold', 0.4)
        self.input_size = kwargs.get('input_size', 640)
        self.num_threads = kwargs.get('num_threads', 3)
        self.target_platform = kwargs.get('target_platform', 'rk3588')
        
        # 更新全局配置
        global OBJ_THRESH, NMS_THRESH
        OBJ_THRESH = self.confidence_threshold
        NMS_THRESH = self.nms_threshold
        
        # 火灾烟雾类别映射
        self.class_names = FIRE_SMOKE_CLASSES
        self.class_names_cn = FIRE_SMOKE_CLASSES_CN
        
        self.pools = None
        self.init_model()
    
    def init_model(self):
        """
        初始化RKNN模型池
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"RKNN模型文件不存在: {self.model_path}")
            
            # 初始化RKNN池
            print(f'🔥 初始化火灾烟雾检测RKNN池...')
            self.pools = FireSmokeRknnPoolExecutor(
                rknn_path=self.model_path, 
                num_thread=self.num_threads, 
                func=fire_smoke_infer
            )
            
            # 预热模型
            for i in range(self.num_threads + 1):
                self.pools.put(np.zeros((640, 640, 3), dtype=np.uint8))
            
            print(f"🔥 RKNN火灾烟雾检测模型初始化完成")
            print(f"模型路径: {self.model_path}")
            print(f"目标平台: {self.target_platform}")
            print(f"线程数: {self.num_threads}")
            print(f"置信度阈值: {self.confidence_threshold}")
            print(f"NMS阈值: {self.nms_threshold}")
            print(f"输入尺寸: {self.input_size}x{self.input_size}")
            
        except Exception as e:
            print(f"RKNN模型初始化失败: {e}")
            if self.pools:
                self.pools.release()
                self.pools = None
            raise e
    
    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        完整推理流程
        Args:
            image: 输入图片 (BGR格式)
        Returns:
            检测结果列表
        """
        try:
            if self.pools is None:
                raise RuntimeError("RKNN模型池未初始化")
            
            # 提交推理任务
            self.pools.put(image)
            
            # 获取推理结果
            detects, flag = self.pools.get()
            
            if not flag:
                return []
            
            return detects if detects else []
            
        except Exception as e:
            print(f"推理失败: {e}")
            return []
    
    def is_fire_emergency(self, detects: List[Dict]) -> tuple:
        """
        判断是否为火灾紧急情况
        Args:
            detects: 检测结果列表
        Returns:
            (is_emergency, max_confidence, emergency_type)
        """
        fire_detected = False
        smoke_detected = False
        max_fire_conf = 0.0
        max_smoke_conf = 0.0
        
        for detect in detects:
            class_name = detect.get('class_name', '')
            conf = detect.get('class_score', 0.0)
            
            if class_name == 'fire':
                fire_detected = True
                max_fire_conf = max(max_fire_conf, conf)
            elif class_name == 'smoke':
                smoke_detected = True
                max_smoke_conf = max(max_smoke_conf, conf)
        
        # 火灾比烟雾更紧急
        if fire_detected:
            return True, max_fire_conf, "fire"
        elif smoke_detected:
            return True, max_smoke_conf, "smoke"
        else:
            return False, 0.0, "normal"
    
    def release(self):
        """释放资源"""
        if self.pools:
            self.pools.release()
            self.pools = None 