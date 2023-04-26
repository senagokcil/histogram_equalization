import cv2
import numpy as np

def img_histogramini_olustur(img, L):
    histogram, bind = np.histogram(img, bins=L, range=(0, L))   # L=256
    return histogram

def duzeltilmis_histogram_olustur(img, L):
    histogram = img_histogramini_olustur(img, L)
    return histogram / img.size

def histogram_degerleri_toplama(duzeltilmis_histogram):
    return np.cumsum(duzeltilmis_histogram)


def histogram_esitleme(img, L):
    duzeltilmis_histogram = duzeltilmis_histogram_olustur(img, L)
    toplanmis_histogram = histogram_degerleri_toplama(duzeltilmis_histogram)
    donusum_fonksiyonu = (L-1) * toplanmis_histogram
    shape = img.shape
    ravel = img.ravel() #matrisi tek satıra döker
    img_kopyasi = np.zeros_like(ravel)
    for i, pixel in enumerate(ravel):
        img_kopyasi[i] = donusum_fonksiyonu[pixel]
    return img_kopyasi.reshape(shape).astype(np.uint8)


def duzeltilmis_histogram_gor():
    L = 2**8
    img = cv2.imread("fotokalitesiz.png", 0)
    img_kopyasi = histogram_esitleme(img, L)

    cv2.imshow("original", img)
    cv2.imshow("histogram_esitleme", img_kopyasi)

duzeltilmis_histogram_gor()

cv2.waitKey(0)
cv2.destroyAllWindows()