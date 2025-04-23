from scipy.signal import wiener, medfilt2d
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import numpy as np
import cv2

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
url = "D:\Code\Python\live_AI\img.jpg"
cam = cv2.VideoCapture(0)
def toRGB(r,g,b):
    frame = np.stack((r, g, b), axis=-1)
    return frame

def wiene(frame, kernel_size = 15):
    r,g,b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    # r,g,b = medfilt2d(r, kernel_size),medfilt2d(g, kernel_size),medfilt2d(b, kernel_size)
    frame = toRGB(r,g,b)
    return frame

def imgenerator(frame):
    prompt = "put a cat on the chair"
    image = PIL.Image.fromarray(np.uint8(frame)).convert('RGB')
    image.save('D:\Code\Python\live_AI' + '\img.jpg')
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1).images
    image = np.array(images[0])
    return image
while True:
    ret, frame = cam.read()
    frame = imgenerator(frame)
    cv2.imshow('Live App', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()