import numpy as np
from PIL import Image
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
import cv2
import os
# def load_image(img_path):
#     img = image.load_img(img_path, target_size=(28, 28))
#     img_tensor = image.img_to_array(img)                    # (height, width, channels)

#     img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
#     img_tensor = img_tensor[:,:,:,:1]
#     # img_tensor = img_tensor.astype('float32')
#     img_tensor = img_tensor / 255.                                      # imshow expects values in the range [0, 1]
#     #print(img_tensor);
#     return img_tensor

model = load_model('fashion_model_dropout.h5py')
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# TEST IMAGE
# img = cv2.imread("images/cero3.png")
# img = cv2.imread("images/cero3.png", cv2.IMREAD_UNCHANGED)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

from PIL import Image

def savePngToJpg(image_path):
    png = Image.open(image_path)
    png.load()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    path, filename = os.path.split(os.path.abspath(image_path))
    finalPath = path + filename.replace('.png', '.jpg')

    background.save(finalPath, 'JPEG', quality=80)
    return finalPath;

# png = Image.open("images/cero3.png")
# png.load() # required for png.split()

# background = Image.new("RGB", png.size, (255, 255, 255))
# background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

# background.save('images/cero3.jpg', 'JPEG', quality=80)
# jpgPath = savePngToJpg("images/2.png")
# print(jpgPath)

# img = cv2.imread("images/6c882ede-b04-28x28.jpg")
# img = np.float32(img)
# path, filename = os.path.split(os.path.abspath(jpgPath))
# newFileName = filename.replace('.jpg', '-28x28.jpg')
# img = cv2.resize(img, (28, 28))
# cv2.imwrite(path + newFileName, img)

img = image.load_img("images/6c882ede-b04-28x28.jpg", target_size=(28, 28), color_mode='grayscale')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.

pred = model.predict(x);
new_predict = np.argmax(np.round(pred),axis=1)
print(new_predict)
