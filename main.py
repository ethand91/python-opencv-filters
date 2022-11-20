import argparse
import numpy as np
import cv2

def vignette(image, level = 2):
    height, width = image.shape[:2]

    x_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    y_resultant_kernel = cv2.getGaussianKernel(height, height/level)

    kernel = y_resultant_kernel * x_resultant_kernel.T
    mask = kernel / kernel.max()

    image_vignette = np.copy(image)

    for i in range(3):
        image_vignette[:,:,i] = image_vignette[:,:,i] * mask

    return image_vignette

def embossed_edges(image):

    kernel = np.array([[0, -3, -3], [3, 0, -3], [3, 3, 0]])

    image_emboss = cv2.filter2D(image, -1, kernel = kernel)

    return image_emboss

def outline(image, k = 9):
    k = max(k, 9)
    kernel = np.array([[-1, -1, -1], [-1, k, -1], [-1, -1, -1]])

    image_outline = cv2.filter2D(image, ddepth = -1, kernel = kernel)

    return image_outline

def style(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0, 0)
    image_style = cv2.stylization(image_blur, sigma_s = 40, sigma_r = 0.1)

    return image_style

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to image file")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    vignette_image = vignette(image)
    embossed_image = embossed_edges(image)
    outline_image = outline(image)
    style_image = style(image)

    cv2.imwrite("vignette.jpg", vignette_image)
    cv2.imwrite("embossed.jpg", embossed_image)
    cv2.imwrite("outline.jpg", outline_image)
    cv2.imwrite("style.jpg", style_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
