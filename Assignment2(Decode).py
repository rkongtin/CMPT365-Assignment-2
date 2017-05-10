from Tkinter import *
from tkFileDialog import askopenfilename
import numpy as np
import cv2


#create a window
app = Tk()
app.title("Window")

#ask to open the *.mrg file
filename1 = askopenfilename()

# --- Decompression ---
#open and read the *.mrg file
no_of_lines = 0
with open('output.mrg', 'r') as f:
    data = f.readlines()
f.close()

#convert strings to floats in each line
line1 = data[0].split()
line2 = data[1].split()
line3 = data[2].split()
line4 = data[3].split()
y_out = np.array(map(float, line1))
u_out = np.array(map(float, line2))
v_out = np.array(map(float, line3))
Q_out = np.array(map(float, line4))
y_quantized = np.reshape(y_out, (-1, 930))
u_quantized = np.reshape(u_out, (-1, 465))
v_quantized = np.reshape(v_out, (-1, 465))
Q = np.reshape(Q_out, (-1, 8))

#do inverse quantization with the quantization matrix
r, c = u_quantized.shape
y_dct = np.zeros((((2*r) + 1), (2*c)), np.float32)
u_dct = np.zeros((r, c), np.float32)
v_dct = np.zeros((r, c), np.float32)
a = 0
b = 0
for i in np.arange(0, ((2*r) + 1)):
    for j in np.arange(0, (2*c)):
        if (i < r and j < c):
            y_dct[i][j] = y_quantized[i][j] * Q[a][b]
            u_dct[i][j] = u_quantized[i][j] * Q[a][b]
            v_dct[i][j] = v_quantized[i][j] * Q[a][b]
        else:
            y_dct[i][j] = y_quantized[i][j] * Q[a][b]
        b += 1
        if (b == 8):
            b = 0
    a += 1
    if (a == 8):
        a = 0

#do 2D IDCT transformation
y = cv2.idct(y_dct)
u_chroma = cv2.idct(u_dct)
v_chroma = cv2.idct(v_dct)

#reverse 4:2:0 chroma subsampling
u = np.zeros((((2*r) + 1), (2*c)), np.float32)
v = np.zeros((((2*r) + 1), (2*c)), np.float32)
for i in np.arange(0, r):
    for j in np.arange(0, c):
        u[(2*i)][(2*j)] = u_chroma[i][j]
        u[(2*i) + 1][(2*j)] = u_chroma[i][j]
        u[(2*i)][(2*j) + 1] = u_chroma[i][j]
        u[(2*i) + 1][(2*j) + 1] = u_chroma[i][j]
        v[(2*i)][(2*j)] = v_chroma[i][j]
        v[(2*i) + 1][(2*j)] = v_chroma[i][j]
        v[(2*i)][(2*j) + 1] = v_chroma[i][j]
        v[(2*i) + 1][(2*j) + 1] = v_chroma[i][j]

#transform from YUV to RGB
img_YUV = np.zeros((((2*r) + 1), (2*c), 3), np.uint8)
img_YUV[:,:,0] = y
img_YUV[:,:,1] = u
img_YUV[:,:,2] = v
image = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
cv2.imshow("Image", image)

app.mainloop()
