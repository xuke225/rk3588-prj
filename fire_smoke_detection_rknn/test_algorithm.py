#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import base64
import cv2
import json


def test_algorithm_api():
    """
    测试算法API接口
    """
    # API地址
    url = "http://localhost:9704/algorithm"
    
    # 读取测试图片（请准备一张测试图片）
    test_image_path = "test_image.jpg"  # 替换为您的测试图片路径
    
    try:
        # 读取图片
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"无法读取图片: {test_image_path}")
            print("请确保图片文件存在，或者使用以下代码生成测试图片：")
            print("  import cv2")
            print("  import numpy as np")
            print("  test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)")
            print("  cv2.imwrite('test_image.jpg', test_img)")
            return
        
        # 将图片编码为base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 构建请求数据
        data = {
            "image_base64": image_base64
        }
        
        # 发送POST请求
        response = requests.post(url, json=data)
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            print("API调用成功!")
            print(f"响应结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"API调用失败! 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")


def create_test_image():
    """
    创建一个测试图片
    """
    import numpy as np
    
    # 创建一个随机图片
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 在图片上绘制一些形状用于测试
    cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.circle(test_img, (300, 150), 50, (255, 0, 0), -1)
    cv2.putText(test_img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存图片
    cv2.imwrite('test_image.jpg', test_img)
    print("测试图片已创建: test_image.jpg")


if __name__ == "__main__":
    import os
    
    # 如果测试图片不存在，创建一个
    if not os.path.exists('test_image.jpg'):
        create_test_image()
    
    # 测试API
    test_algorithm_api() 