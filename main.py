from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from imageProcessor import ImageProcessor
import base64

app = FastAPI()
processor = ImageProcessor()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# def read_image(image_bytes: bytes) -> np.ndarray:
#     """Convert image bytes to OpenCV format"""
#     image = np.asarray(bytearray(image_bytes), dtype=np.uint8)
#     return cv2.imdecode(image, cv2.IMREAD_COLOR)

def read_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes to OpenCV format"""
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

@app.post("/compare")
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        # Read images into OpenCV format
        img1 = read_image(await image1.read())
        img2 = read_image(await image2.read())
        
        if img1 is None or img2 is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        print('shape: ', img1.shape)
        print('shape: ', img2.shape)
        # if img1.shape != img2.shape:
        #     return JSONResponse(content={"error": "Images must have the same dimensions"}, status_code=400)
        
        processor.clean_image = img1
        processor.dirty_image = img2

        (image_list, lab_list)= processor.process()
        print(1)
        print(len(image_list))
        image_byte_list = []
        label_list = []

        for i in range(len(image_list)):
            image = image_list[i]
            lab = lab_list[i]
            if image is None:
                continue

            _, img_encoded = cv2.imencode(".png", image)
            image_byte_list.append(base64.b64encode(img_encoded).decode('utf-8'))
            label_list.append(lab)
        
            
        return JSONResponse(content={"images": image_byte_list, "labels": label_list})
        # return JSONResponse(content={"images": "ok", "labels": "ok"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
