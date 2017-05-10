from Tkinter import *
from tkFileDialog import askopenfilename
import numpy as np
import cv2


#display the beach image
def beach():
    cv2.imshow("Image", beach_image)

#display the sprite image
def sprite():
    cv2.imshow("Image", sprite_image)

#display the merged image
def merged():
    cv2.imshow("Image", merged_image)


#create a window
app = Tk()
app.title("Window")

#ask to open the images
filename1 = askopenfilename()
filename2 = askopenfilename()

#read the images
beach_image = cv2.imread(filename1)
sprite_image = cv2.imread(filename2)

#get the size of the sprite image to place on the beach image
merged_image = cv2.imread(filename1)
rows, cols, channels = sprite_image.shape
merged_part = beach_image[200:rows+200, 200:cols+200]

#make a mask of the sprite image and the inverse of the mask
bgr2gray = cv2.cvtColor(sprite_image, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(bgr2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#make the background and foreground
background = cv2.bitwise_and(merged_part, merged_part, mask = mask_inv)
foreground = cv2.bitwise_and(sprite_image, sprite_image, mask = mask)

#merge the background and foreground
merging = cv2.add(background, foreground)
merged_image[200:rows+200, 200:cols+200] = merging
cv2.imwrite("Merged.png", merged_image)


# --- Compression ---
#transform from RGB to YUV
img_YUV = cv2.cvtColor(merged_image, cv2.COLOR_BGR2YUV)
rows, cols, channels = img_YUV.shape
y = np.zeros((rows, cols), np.float32)
u = np.zeros((rows, cols), np.float32)
v = np.zeros((rows, cols), np.float32)
y[:rows, :cols] = img_YUV[:,:,0]
u[:rows, :cols] = img_YUV[:,:,1]
v[:rows, :cols] = img_YUV[:,:,2]

#do 4:2:0 chroma subsampling
u_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
v_chroma = np.zeros(((rows/2), (cols/2)), np.float32)
for i in np.arange(0, (rows/2)):
    for j in np.arange(0, (cols/2)):
        u_chroma[i][j] = (u[(2*i)][(2*j)] + u[(2*i) + 1][(2*j)] + u[(2*i)][(2*j) + 1] + u[(2*i) + 1][(2*j) + 1])/4
        v_chroma[i][j] = (v[(2*i)][(2*j)] + v[(2*i) + 1][(2*j)] + v[(2*i)][(2*j) + 1] + v[(2*i) + 1][(2*j) + 1])/4

#do 2D DCT transformation
y_dct = cv2.dct(y)
u_dct = cv2.dct(u_chroma)
v_dct = cv2.dct(v_chroma)

#do quantization with the quantization matrix
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])
y_quantized = np.zeros((rows, cols), np.float32)
u_quantized = np.zeros(((rows/2), (cols/2)), np.float32)
v_quantized = np.zeros(((rows/2), (cols/2)), np.float32)
a = 0
b = 0
for i in np.arange(0, rows):
    for j in np.arange(0, cols):
        if (i < ((rows/2)) and j < ((cols/2))):
            y_quantized[i][j] = y_dct[i][j]/Q[a][b]
            u_quantized[i][j] = u_dct[i][j]/Q[a][b]
            v_quantized[i][j] = v_dct[i][j]/Q[a][b]
        else:
            y_quantized[i][j] = y_dct[i][j]/Q[a][b]
        b += 1
        if (b == 8):
            b = 0
    a += 1
    if (a == 8):
        a = 0

#scan the output matrix of quantization to make it into 1D array
y_out = y_quantized.ravel()
u_out = u_quantized.ravel()
v_out = v_quantized.ravel()

#scan the quantization matrix to make it into 1D array
Q = Q.ravel()

#concatenate it with quantization matrix
output_y = ["%.10f" % x for x in y_out]
output_u = ["%.10f" % x for x in u_out]
output_v = ["%.10f" % x for x in v_out]
output_Q = ["%.0f" % x for x in Q]
size_y = len(output_y)
size_u = len(output_u)
size_v = len(output_v)
size_Q = len(output_Q)

#save it into *.mrg file format
f = open('output.mrg', 'w+')
for i in np.arange(0, size_y - 1):
    f.write(output_y[i])
    f.write(" ")
f.write(output_y[size_y - 1])
f.write("\n")
for i in np.arange(0, size_u - 1):
    f.write(output_u[i])
    f.write(" ")
f.write(output_u[size_u - 1])
f.write("\n")
for i in np.arange(0, size_v - 1):
    f.write(output_v[i])
    f.write(" ")
f.write(output_v[size_v - 1])
f.write("\n")
for i in np.arange(0, size_Q - 1):
    f.write(output_Q[i])
    f.write(" ")
f.write(output_Q[size_Q - 1])
f.close()


beach_button = Button(app, text="Display Beach Image", command=beach)
sprite_button = Button(app, text="Display Sprite Image", command=sprite)
merged_button = Button(app, text="Display The Merged Image", command=merged)

beach_button.pack()
sprite_button.pack()
merged_button.pack()

app.mainloop()
