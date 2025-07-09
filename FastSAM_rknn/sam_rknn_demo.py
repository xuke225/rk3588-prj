import os
import time
import sys
import numpy as np
import random
import cv2
from rknn.api import RKNN
from math import exp

# RKNN模型路径
RKNN_MODEL = './models/FastSAM_S.rknn'

meshgrid = []
class_num = 1
head_num = 3
strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]
nms_thresh = 0.45
object_thresh = 0.25

input_imgH = 640
input_imgW = 640
mask_num = 32
dfl_num = 16


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, mask):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.mask = mask


def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin
    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0
    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    total = area1 + area2 - innerArea
    return innerArea / total


def NMS(detectResult):
    predBoxs = []
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs)):
                if sort_detectboxs[i].classId == sort_detectboxs[j].classId:
                    iou = IOU(sort_detectboxs[i].xmin, sort_detectboxs[i].ymin,
                              sort_detectboxs[i].xmax, sort_detectboxs[i].ymax,
                              sort_detectboxs[j].xmin, sort_detectboxs[j].ymin,
                              sort_detectboxs[j].xmax, sort_detectboxs[j].ymax)
                    if iou > nms_thresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')
    detectResult = []
    output = [o.reshape(-1) for o in out]
    scale_h, scale_w = img_h / input_imgH, img_w / input_imgW
    gridIndex = -2

    for index in range(head_num):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]
        msk = output[head_num * 2 + index]

        for h in range(map_size[index][0]):
            for w in range(map_size[index][1]):
                gridIndex += 2
                cls_max = sigmoid(cls[h * map_size[index][1] + w])
                if cls_max > object_thresh:
                    regdfl = []
                    for lc in range(4):
                        sfsum = sum(exp(reg[((lc * dfl_num) + df) * map_size[index][0] * map_size[index][1] + h *
                                            map_size[index][1] + w])
                                    for df in range(dfl_num))
                        locval = sum((exp(reg[((lc * dfl_num) + df) * map_size[index][0] * map_size[index][1] + h *
                                              map_size[index][1] + w]) / sfsum) * df
                                     for df in range(dfl_num))
                        regdfl.append(locval)

                    x1 = (meshgrid[gridIndex] - regdfl[0]) * strides[index]
                    y1 = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index]
                    x2 = (meshgrid[gridIndex] + regdfl[2]) * strides[index]
                    y2 = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index]
                    xmin = max(0, x1 * scale_w)
                    ymin = max(0, y1 * scale_h)
                    xmax = min(img_w, x2 * scale_w)
                    ymax = min(img_h, y2 * scale_h)
                    mask = [msk[m * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w]
                            for m in range(mask_num)]
                    detectResult.append(DetectBox(0, cls_max, xmin, ymin, xmax, ymax, mask))

    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    return predBox


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def seg_postprocess(out, predbox, img_h, img_w):
    print('seg_postprocess ... ')
    protos = np.array(out[9][0])
    c, mh, mw = protos.shape
    seg_mask = np.zeros((mh, mw, 3), dtype=np.uint8)
    mask_contour = []

    for box in predbox:
        masks_in = np.array(box.mask).reshape(-1, c)
        masks = 1 / (1 + np.exp(-masks_in @ protos.reshape(c, -1)))
        masks = masks.reshape(mh, mw)
        xmin = int(box.xmin / img_w * mw + 0.5)
        ymin = int(box.ymin / img_h * mh + 0.5)
        xmax = int(box.xmax / img_w * mw + 0.5)
        ymax = int(box.ymax / img_h * mh + 0.5)

        gray_mask = np.zeros((mh, mw), dtype=np.uint8)  # 修复为二维数组
        gray_mask[ymin:ymax, xmin:xmax] = (masks[ymin:ymax, xmin:xmax] > 0.5).astype(np.uint8) * 255

        gray_mask = cv2.resize(gray_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_contour.append(contours)

    seg_mask = cv2.resize(seg_mask, (img_w, img_h))
    return seg_mask, mask_contour


def rknn_inference(img):
    # 创建RKNN实例
    rknn = RKNN(verbose=True)
    
    # 配置RKNN，指定目标平台
    print('--> Config RKNN model')
    rknn.config(target_platform='rk3588')
    
    # 直接加载RKNN模型
    print('--> Loading RKNN model: ', RKNN_MODEL)
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        return None, 0
    
    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588')  # 明确指定目标平台
    if ret != 0:
        print('Init runtime environment failed')
        return None, 0
    print('--> Runtime initialized successfully')

    # 执行推理并计时
    inference_start = time.time()
    outputs = rknn.inference(inputs=[img])
    rknn_inference_time = time.time() - inference_start
    print(f"[TIME] 纯RKNN推理时间: {rknn_inference_time:.4f} 秒 （核心指标）")

    # 释放资源
    rknn.release()
    return outputs, rknn_inference_time


if __name__ == '__main__':
    print('This is main ...')
    start_total = time.time()  # 总时间计时开始

    # 1. 网格初始化（可选计时）
    GenerateMeshgrid_start = time.time()
    GenerateMeshgrid()
    print(f"[TIME] 网格初始化耗时: {time.time() - GenerateMeshgrid_start:.4f} 秒")

    # 2. 图像预处理
    img_path = './data/test.jpg'
    origin_image = cv2.imread(img_path)
    if origin_image is None:
        print(f"错误：无法加载图像 {img_path}")
        sys.exit(1)
        
    img_h, img_w = origin_image.shape[:2]
    preprocess_start = time.time()
    img = cv2.resize(origin_image, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    preprocess_time = time.time() - preprocess_start
    print(f"[TIME] 预处理耗时: {preprocess_time:.4f} 秒")

    # 3. RKNN推理（重点计时部分）
    output, rknn_time = rknn_inference(img)
    if output is None:
        print("RKNN推理失败，退出程序")
        sys.exit(1)

    # 4. 后处理计时
    postprocess_start = time.time()
    out = [o for o in output]
    predbox = postprocess(out, img_h, img_w)
    mask, mask_contour = seg_postprocess(out, predbox, img_h, img_w)
    postprocess_time = time.time() - postprocess_start
    print(f"[TIME] 后处理耗时: {postprocess_time:.4f} 秒")

    # 5. 结果绘制
    result_img = cv2.addWeighted(origin_image, 0.8, mask, 0.2, 0.0)
    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        cv2.drawContours(result_img, mask_contour[i], -1, random_color(), 2)
    cv2.imwrite(r'./data/result.jpg', result_img)

    # 6. 总时间统计
    total_time = time.time() - start_total
    print("\n==================== 性能统计 ====================")
    print(f"[核心] 纯RKNN推理时间:        {rknn_time:.4f} 秒")
    print(f"[阶段] 预处理时间:            {preprocess_time:.4f} 秒")
    print(f"[阶段] 后处理时间:            {postprocess_time:.4f} 秒")
    print(f"[总耗时] 全流程时间:          {total_time:.4f} 秒")
    print(f"[推理占比] 推理/总时间:        {rknn_time / total_time * 100:.2f}%")
    print("====================================================")

    print('Finished!')
