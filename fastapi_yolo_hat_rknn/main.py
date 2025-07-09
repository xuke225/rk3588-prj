from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import cv2

# 引入检测逻辑
from yolo import infer, rknnPoolExecutor

app = FastAPI()

NUM_THREAD=3
# 初始化 RKNN 池
pools = rknnPoolExecutor(rknn_path="./model/best.rknn", num_thread=NUM_THREAD, func=infer)

for i in range(NUM_THREAD + 1):
    pools.put(np.zeros((640, 640, 3), dtype=np.uint8))

@app.get("/")
async def index():
    return {"message": "index"}

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/algorithm")
async def algorithm(request: ImageRequest):
    image_base64 = request.image_base64
    if not image_base64:
        raise HTTPException(status_code=400, detail="Image base64 is required")

    encoded_image_byte = base64.b64decode(image_base64)
    image_array = np.frombuffer(encoded_image_byte, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    pools.put(image)
    detects, flag = pools.get()

    happen = False
    happenScore = 0.0
    if len(detects) > 0:
        happen = True
        happenScore = 1.0

    result = {
        "happen": happen,
        "happenScore": happenScore,
        "detects": detects
    }
    return JSONResponse(content={"code": 1000, "msg": "success", "result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9703)
