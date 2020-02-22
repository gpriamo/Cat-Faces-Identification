import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print(cv.__version__)
img = cv.imread('../images/random/Cat03.jpg')
px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)

print(img.size)
print(img.dtype)

img2 = cv.imread('../images/random/prince-cay-01.jpg')
img2grey = cv.imread('../images/random/prince-cay-01.jpg', 0)
cv.imshow('res',img2grey)
plt.hist(img2grey.ravel(),256,[0,256])
plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

#cv.waitKey(0)
#cv.destroyAllWindows()

f = np.fft.fft2(img2grey)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img2grey, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

dft = cv.dft(np.float32(img2grey),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img2grey, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img[0:rows, 0:cols]
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)
# # Now black-out the area of logo in ROI
# img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# # Take only region of logo from logo image.
# img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# # Put logo in ROI and modify the main image
# dst = cv.add(img1_bg,img2_fg)
# img[0:rows, 0:cols ] = dst
# cv.imshow('res',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
