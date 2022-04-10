
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import argparse

def detectARCode(image, SavePath):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = blurFFT(image_gray, SavePath)
    image_edge = edgeFindFFT(image_gray, SavePath)
    return image_blur, image_edge

def blurFFT(gray_image, SavePath):
    image_gray = gray_image.copy()
    #fft for blurring
    fft_blur = scipy.fft.fft2(image_gray, axes = (0,1))
    fft_shifted_blur = scipy.fft.fftshift(fft_blur)
    magnitude_spectrum_fft_shifted_blur = 20*np.log(np.abs(fft_shifted_blur))

    #fft+mask
    fft_masked_blur = fft_shifted_blur * GaussianMask(image_gray.shape, 30, 30)
    magnitude_spectrum_masked_blur = 20*np.log(np.abs(fft_masked_blur))

    #image back
    img_back_blur = scipy.fft.ifftshift(fft_masked_blur)
    img_back_blur = scipy.fft.ifft2(img_back_blur)
    img_back_blur = np.abs(img_back_blur)

    fx, plts = plt.subplots(2,2,figsize = (15, 10))
    plts[0][0].imshow(image_gray, cmap = 'gray')
    plts[0][0].set_title('Gray Image')
    plts[0][1].imshow(magnitude_spectrum_fft_shifted_blur, cmap = 'gray')
    plts[0][1].set_title('FFT of Gray Image')
    plts[1][0].imshow(magnitude_spectrum_masked_blur, cmap = 'gray')
    plts[1][0].set_title('Mask + FFT of Gray Image')
    plts[1][1].imshow(img_back_blur, cmap = 'gray')
    plts[1][1].set_title('Blurred Image')
    plt.savefig(SavePath + "fft_blur" + ".jpg")

    return img_back_blur

def edgeFindFFT(image, SavePath):
    thresh = image.copy()
    #fft
    fft_edges = scipy.fft.fft2(thresh, axes = (0,1))
    fft_shifted_edges = scipy.fft.fftshift(fft_edges)
    magnitude_spectrum_fft_shifted_edges = 20*np.log(np.abs(fft_shifted_edges))

    #fft+mask
    cmask = CircularMask(thresh.shape, 15, True)
    fft_masked_edge = fft_shifted_edges * CircularMask(thresh.shape, 100, True)
    magnitude_spectrum_masked_edge = 20*np.log(np.abs(fft_masked_edge))

    #image back
    img_back_edge = scipy.fft.ifftshift(fft_masked_edge)
    img_back_edge = scipy.fft.ifft2(img_back_edge)
    img_back_edge = np.abs(img_back_edge)

    fx, plts = plt.subplots(2,2,figsize = (15,10))
    plts[0][0].imshow(thresh, cmap = 'gray')
    plts[0][0].set_title('Thresholded Image')
    plts[0][1].imshow(magnitude_spectrum_fft_shifted_edges, cmap = 'gray')
    plts[0][1].set_title('FFT of Thresholded Image')
    plts[1][0].imshow(magnitude_spectrum_masked_edge, cmap = 'gray')
    plts[1][0].set_title('Mask + FFT of Thresholded Image')
    plts[1][1].imshow(img_back_edge, cmap = 'gray')
    plts[1][1].set_title('Edges of tag')
    plt.savefig(SavePath + "fft_edge"  + ".jpg")

    return img_back_edge

def CircularMask(image_size, radius, high_pass = True):
    rows, cols = image_size
    centre_x, centre_y = int(rows / 2), int(cols / 2)
    center = [centre_x, centre_y]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius*radius

    if high_pass:
        mask = np.ones((rows, cols)) 
        mask[mask_area] = 0
    else:
        mask = np.zeros((rows, cols)) 
        mask[mask_area] = 1

    return mask

def GaussianMask(image_size, sigma_x, sigma_y):
    cols, rows = image_size
    centre_x, centre_y = rows / 2, cols / 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(x,y)
    mask = np.exp(-(np.square((X - centre_x)/sigma_x) + np.square((Y - centre_y)/sigma_y)))
    return mask

def main():
    VideoFilePath = "/home/prateek/ENPM673/Project1/1tagvideo.mp4"
    SavePath = "/home/prateek/ENPM673/Project1/Prob1A_Output/"
    RefTagFileName = "ref_marker.png"
    ref_tag_image = cv2.imread(RefTagFileName)

    cap = cv2.VideoCapture(VideoFilePath)
    frame_index = 302

    i = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Stream ended..")
            break
        i = i + 1
        if i == frame_index:
            chosen_frame = frame

    cap.release()
    cv2.destroyAllWindows()
    i = frame_index

    image = chosen_frame.copy()
    tag_blur, tag_edge = detectARCode(image, SavePath)

    print("The results are saved in the SavePath.")

if __name__ == '__main__':
    main()