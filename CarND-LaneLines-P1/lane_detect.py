#importing some useful packages
#import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import math

# heyye uj

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def average_slope_intercept(lines):
    # This function finds the slope and intercept of the left and right lane
    left_lines = [] # slope, intercept
    right_lines = [] # slope, intercept
    left_weights = [] # length
    right_weights = [] # length

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1==x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope* x1)
            length = np.sqrt(((y2 - y1)**2) + ((x2 - x1)**2))
            if slope<0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights)>0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_point (y1, y2, line_slope_intercept):
    # This function converts the slope and intercept of each line into pixel points
    if line_slope_intercept is None:
        return None
    slope, intercept = line_slope_intercept
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    # This function creates a full line from pixel points
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_point(y1, y2, left_lane)
    right_line = pixel_point(y1, y2, right_lane)
    return left_line, right_line

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_image = np.zeros_like(img)
    for line in lines:
        if line is not None:
            #for x1, y1, x2, y2 in line:
                cv2.line(img, *line, color, thickness)
    return weighted_img(img=img, initial_img=line_image)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lane_lines(line_img, lines))
    return lines


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

img_list = os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
for img in img_list:

    image = mpimg.imread("test_images/" + img)

    gray_img = grayscale(image)

    blur_img = gaussian_blur(gray_img, 3)

    canny_img = canny(blur_img, 50, 150)

    img_shape = image.shape
    vertices = np.array([[(0.51 * img_shape[1], 0.58 * img_shape[0]), (0.49 * img_shape[1], 0.58 * img_shape[0]),
                          (0, img_shape[0]), (img_shape[1], img_shape[0])]], dtype=np.int32)
    cropped_img = region_of_interest(canny_img, vertices)

    line = hough_lines(cropped_img, rho=1, theta=np.pi / 180, threshold=35, min_line_len=5, max_line_gap=2)

    #lines_img = weighted_img(img=line, initial_img=image, α=0.8, β=1., γ=0.)

    connected_lines_img = draw_lines(img=image, lines=lane_lines(image, line) )

    result = connected_lines_img
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_images_output/" + img, result)
    plt.imshow(result)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    gray_img = grayscale(image)

    blur_img = gaussian_blur(gray_img, 3)

    canny_img = canny(blur_img, 50, 150)

    img_shape = image.shape
    vertices = np.array([[(0.51 * img_shape[1], 0.58 * img_shape[0]), (0.49 * img_shape[1], 0.58 * img_shape[0]),
                          (0, img_shape[0]), (img_shape[1], img_shape[0])]], dtype=np.int32)
    cropped_img = region_of_interest(canny_img, vertices)

    line = hough_lines(cropped_img, rho=1, theta=np.pi / 180, threshold=35, min_line_len=5, max_line_gap=2)

    #lines_img = weighted_img(img=line, initial_img=image, α=0.8, β=1., γ=0.)

    connected_lines_img = draw_lines(img=image, lines=lane_lines(image, line) )

    result = connected_lines_img

    return result

# white_output = 'test_videos_output/solidWhiteRight.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# #%time
# white_clip.write_videofile(white_output, audio=False)


# yellow_output = 'test_videos_output/solidYellowLeft.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
# clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# #%time
# yellow_clip.write_videofile(yellow_output, audio=False)


# challenge_output = 'test_videos_output/challenge.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
# clip3 = VideoFileClip('test_videos/challenge.mp4')
# challenge_clip = clip3.fl_image(process_image)
# #%time
# challenge_clip.write_videofile(challenge_output, audio=False)