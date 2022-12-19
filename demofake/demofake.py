import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
img=cv2.imread('gan.png',flags=0)
cv2.imshow("1",img)
cv2.waitKey()
M=img.shape[0]
N=img.shape[1]
fft=np.fft.fft2(img)
fftshift=np.fft.fftshift(fft)
redcut=100
greencut=200
bluecut=150
bluecenter=150
bluewidth=100
blueu0=10
bluev0=10
D=np.zeros([M,N])
RedH=np.zeros([M,N])
GreenH=np.zeros([M,N])
BlueD=np.zeros([M,N])
BlueH=np.zeros([M,N])
Out=np.zeros([M,N,3])
for u in range(1,M):
    for v in range(1,N):
        D[u][v]=math.sqrt(u**2+v**2)
        RedH[u][v]=1/(1+(math.sqrt(2)-1)*(D[u][v]/redcut)**2)#红色滤波器为低通
        GreenH[u][v]=1/(1+(math.sqrt(2)-1)*(greencut/D[u][v])**2)#绿色滤波器为高通
        # BlueD[u][v]=math.sqrt((u-blueu0)**2+(v-bluev0)**2)
        # BlueH[u][v]=1-1/(1+BlueD[u][v]*bluewidth/((BlueD[u][v]**2-(bluecenter)**2)**2))
        BlueH[u][v] = 1/(1+(math.sqrt(2)-1)*(bluecut/D[u][v])**2)#蓝色滤波器为高通

Red=RedH*fft
Redcolor=np.fft.ifft2(Red)
Green=GreenH*fft
Greencolor=np.fft.ifft2(Green)
Blue=BlueH*fft
Bluecolor=np.fft.ifft2(Blue)
Redcolor=Redcolor.real/256
Greencolor=Greencolor.real/256
Bluecolor=Bluecolor.real/256
for i in range(1,M):
    for j in range(1,N):
        Out[i][j][2]=Redcolor[i][j]
        Out[i][j][1]=Greencolor[i][j]
        Out[i][j][0]=Bluecolor[i][j]
Out=abs(Out)

# Out= cv2.resize(Out, (1454,813))

cv2.imshow("2",Out)
cv2.waitKey()
# f,ax=plt.subplots(1,2)
# ax[0].imshow(img)
# ax[1].imshow(Out)
# plt.show()