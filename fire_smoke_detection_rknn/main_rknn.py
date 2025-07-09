from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import cv2
import logging
import os
from typing import Dict, Any

# 根据模型文件类型导入不同的算法类
try:
    # 检查模型文件类型
    model_path = os.environ.get('FIRE_MODEL_PATH', '/home/orangepi/EagleCV/Algorithms/fire_smoke_detection/model/yolo_fire.rknn')
    
    if model_path.endswith('.rknn'):
        print("🔥 检测到RKNN模型，使用RKNN推理引擎")
        from algorithm_rknn import FireSmokeDetectorRKNN as FireSmokeDetector
        USE_RKNN = True
    else:
        print("🔥 检测到PyTorch模型，使用标准推理引擎")
        from algorithm import FireSmokeDetector
        USE_RKNN = False
        
except ImportError as e:
    print(f"导入算法模块失败: {e}")
    print("回退到标准算法模块")
    from algorithm import FireSmokeDetector
    USE_RKNN = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="火灾烟雾检测 API (RKNN优化版)", 
    version="1.1.0",
    description="基于YOLOv11的火灾烟雾实时检测系统，支持RKNN加速"
)

# 初始化算法实例
try:
    if USE_RKNN:
        # RKNN模型配置
        algorithm = FireSmokeDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            nms_threshold=0.4,
            num_threads=3,
            target_platform='rk3588'
        )
    else:
        # PyTorch模型配置
        algorithm = FireSmokeDetector(
            model_path="Algorithms/fire_smoke_detection/model/yolo-fire.pt",
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
    
    logger.info(f"火灾烟雾检测算法初始化成功 (引擎: {'RKNN' if USE_RKNN else 'PyTorch'})")
except Exception as e:
    logger.error(f"算法初始化失败: {e}")
    algorithm = None

class ImageRequest(BaseModel):
    image_base64: str

@app.get("/")
async def index():
    """健康检查接口"""
    engine = "RKNN" if USE_RKNN else "PyTorch"
    return {
        "message": "🔥 火灾烟雾检测API服务运行中",
        "engine": engine,
        "status": "healthy" if algorithm else "error"
    }

@app.get("/health")
async def health_check():
    """详细健康检查"""
    engine = "RKNN" if USE_RKNN else "PyTorch"
    status = {
        "service": "火灾烟雾检测API",
        "version": "1.1.0",
        "engine": engine,
        "model_loaded": algorithm is not None,
        "timestamp": None
    }
    
    if algorithm:
        try:
            # 测试模型是否正常工作
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = algorithm.infer(test_image)
            status["model_status"] = "正常"
            status["last_test"] = "通过"
        except Exception as e:
            status["model_status"] = f"异常: {str(e)}"
            status["last_test"] = "失败"
    else:
        status["model_status"] = "未加载"
        status["last_test"] = "未测试"
    
    return JSONResponse(content=status)

@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    if not algorithm:
        raise HTTPException(status_code=500, detail="算法未初始化")
    
    info = {
        "engine": "RKNN" if USE_RKNN else "PyTorch",
        "model_path": getattr(algorithm, 'model_path', 'unknown'),
        "confidence_threshold": getattr(algorithm, 'confidence_threshold', 0.5),
        "nms_threshold": getattr(algorithm, 'nms_threshold', 0.4),
        "input_size": getattr(algorithm, 'input_size', 640),
        "class_names": getattr(algorithm, 'class_names', {}),
        "class_names_cn": getattr(algorithm, 'class_names_cn', {}),
    }
    
    if USE_RKNN:
        info.update({
            "num_threads": getattr(algorithm, 'num_threads', 3),
            "target_platform": getattr(algorithm, 'target_platform', 'rk3588')
        })
    
    return JSONResponse(content=info)

@app.post("/algorithm")
async def algorithm_endpoint(request: ImageRequest):
    """标准算法接口（兼容EagleCV系统）"""
    return await detect_fire_smoke(request)

@app.post("/detect")
async def detect_fire_smoke(request: ImageRequest):
    """火灾烟雾检测专用接口"""
    if not algorithm:
        raise HTTPException(status_code=500, detail="算法未初始化")
    
    try:
        # 解码base64图像
        image_base64 = request.image_base64
        if not image_base64:
            raise HTTPException(status_code=400, detail="图像base64数据为空")

        encoded_image_byte = base64.b64decode(image_base64)
        image_array = np.frombuffer(encoded_image_byte, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像数据")

        # 执行火灾烟雾检测
        detects = algorithm.infer(image)
        
        # 计算基本信息
        happen = len(detects) > 0
        happenScore = max([d.get('class_score', 0.0) for d in detects]) if detects else 0.0
        
        # 紧急状态分析
        is_emergency, max_confidence, emergency_type = algorithm.is_fire_emergency(detects)
        
        # 确定紧急程度
        if emergency_type == "fire":
            emergency_level = "high" if max_confidence > 0.8 else "medium"
        elif emergency_type == "smoke":
            emergency_level = "medium" if max_confidence > 0.7 else "low"
        else:
            emergency_level = "low"
        
        # 构建响应结果
        result = {
            "happen": happen,
            "happenScore": happenScore,
            "detects": detects,
            "emergency": {
                "is_emergency": is_emergency,
                "emergency_type": emergency_type,
                "emergency_level": emergency_level,
                "max_confidence": max_confidence
            }
        }
        
        return JSONResponse(content={
            "code": 1000, 
            "msg": "success", 
            "result": result,
            "engine": "RKNN" if USE_RKNN else "PyTorch"
        })
        
    except Exception as e:
        logger.error(f"检测过程出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 5000, 
                "msg": f"检测失败: {str(e)}", 
                "result": None
            }
        )

# 优雅关闭处理
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    if algorithm and hasattr(algorithm, 'release'):
        try:
            algorithm.release()
            logger.info("算法资源已释放")
        except Exception as e:
            logger.error(f"释放算法资源时出错: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9704)