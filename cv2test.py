from PIL import Image
import numpy as np
import os

path = "C:\\Users\\19586\\OneDrive - devprojs\\图片\\送你一朵小花.png"
img = Image.open(path).convert('L')
img.show()

