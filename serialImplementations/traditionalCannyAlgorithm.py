# libraries used
import argparse
import cv2

# argument handling
ap = argparse.ArgumentParser(
        description='Takes an image and detects the edges using Canny Edge Detection Algorithm')
ap.add_argument("-i", "--image",
        required=True, help="an image to detect the edges for")
ap.add_argument("-s", "--sigma",
        required=True, type=float, help="the sigma value to use")
ap.add_argument("-lt", "--low", required=True, type=int,
        help="low threshhold value")
ap.add_argument("-ht", "--high", required=True, type=int,
        help="high threshhold value")
args = vars(ap.parse_args())

# My implementation of Canny Edge Detection
# Step 1: apply gaussian blur
# Step 2: calculate gradients x and y
# Step 3: non-Maximum supression
# Step 4: Hysteresis Thresholding
def edgeDetection(image, sigma, low, high):
    #Step 1: Gaussian
    blurred = cv2.GaussianBlur(image, (5,5), sigma)
    
    #Step 2: Gradient Calculation
    # Apply sobel filter
    xGradient = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    yGradient = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    # convert the xGradient and yGradient to polar coordinates
    magnitude, angle = cv2.cartToPolar(xGradient, yGradient, angleInDegrees = True)

    #Step 3: non-Maximum suppression
    # loop through all pixels in the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # calculate the neighbor pixels of the current pixel 
            # based on angle
            currAngle = angle[y,x]
            if (currAngle > 180):
                currAngle = currAngle-180
            if (currAngle <= 22.5 or currAngle > 157.5):
                firstNeighborX = x - 1
                firstNeighborY = y
                secondNeighborX = x + 1
                secondNeighborY = y
            elif (currAngle > 22.5 and currAngle <= 67.5):
                firstNeighborX = x - 1
                firstNeighborY = y - 1
                secondNeighborX = x + 1
                secondNeighborY = y + 1
            elif (currAngle > 67.5 and currAngle <= 112.5):
                firstNeighborX = x - 1
                firstNeighborY = y + 1
                secondNeighborX = x + 1
                secondNeighborY = y - 1
            elif (currAngle > 112.5 and currAngle <= 157.5):
                firstNeighborX = x
                firstNeighborY = y + 1
                secondNeighborX = x
                secondNeighborY = y - 1
            # check each pixel against its neighbor
            # set the magnitude of the pixel to 0
            # if either neighbor has a larger magnitude
            if (firstNeighborX >= 0 and firstNeighborY >= 0):
                if (image.shape[1] > firstNeighborX and image.shape[0] > firstNeighborY ):
                    if (magnitude[y,x] < magnitude[firstNeighborY, firstNeighborX]):
                        magnitude[y,x] = 0
            if (secondNeighborX >= 0 and secondNeighborY >= 0):
                if (image.shape[1] > secondNeighborX and image.shape[0] > secondNeighborY ):
                    if (magnitude[y,x] < magnitude[secondNeighborY, secondNeighborX]):
                        magnitude[y,x] = 0

    # Step 4: hysteresis threshold
    # loop through all pixels in the image
    # 0 = discard
    # 100 = weak threshold
    # 255 = strong threshold
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            currMagnitude = magnitude[y,x]

            if currMagnitude < low:
                magnitude[y,x] = 0
            elif currMagnitude >= low and currMagnitude < high:
                magnitude[y,x] = 100
            else:
                magnitude[y,x] = 255
    # loop through and determine if weak thresholds
    # should be discarded
    # from top left first
    top_left = magnitude.copy()
    for x in range(image.shape[1] - 1):
        for y in range(image.shape[0] - 1):
            # weak thresholds defined with 100
            if (top_left[y,x] == 100):
                # check each neighbor of the weak threshold
                try:
                    if (top_left[y+1,x] == 255 
                            or top_left[y-1, x] == 255 
                            or top_left[y+1, x-1] == 255 
                            or top_left[y+1, x+1] == 255
                            or top_left[y-1, x-1] == 255 
                            or top_left[y-1, x+1] == 255 
                            or top_left[y, x - 1] == 255 
                            or top_left[y, x + 1] == 255):
                        top_left[y,x] = 255
                    else:
                        top_left[y,x] = 0
                except IndexError as e:
                    pass
    # from top right next
    top_right = magnitude.copy()
    for x in range(image.shape[1]-1)[::-1]:
        for y in range(image.shape[0]-1):
            if (top_right[y,x] == 100):
                try:
                    if(top_right[y+1,x] == 255 
                            or top_right[y-1, x] == 255 
                            or top_right[y+1, x-1] == 255 
                            or top_right[y+1, x+1] == 255
                            or top_right[y-1, x-1] == 255 
                            or top_right[y-1, x+1] == 255 
                            or top_right[y, x - 1] == 255 
                            or top_right[y, x + 1] == 255):
                        top_right[y,x] = 255
                    else:
                        top_right[y,x] = 0
                except IndexError as e:
                    pass
    # from bottom left
    bottom_left = magnitude.copy()
    for x in range(image.shape[1]-1):
        for y in range(image.shape[0]-1)[::-1]:
            if (bottom_left[y,x] == 100):
                try:
                    if(bottom_left[y+1,x] == 255 
                            or bottom_left[y-1, x] == 255 
                            or bottom_left[y+1, x-1] == 255 
                            or bottom_left[y+1, x+1] == 255
                            or bottom_left[y-1, x-1] == 255 
                            or bottom_left[y-1, x+1] == 255 
                            or bottom_left[y, x - 1] == 255 
                            or bottom_left[y, x + 1] == 255):
                        bottom_left[y,x] = 255
                    else:
                        bottom_left[y,x] = 0
                except IndexError as e:
                    pass
    # from bottom right
    bottom_right = magnitude.copy()
    for x in range(image.shape[1]-1)[::1]:
        for y in range(image.shape[0]-1)[::-1]:
            if (bottom_right[y,x] == 100):
                try:
                    if (bottom_right[y+1,x] == 255 
                            or bottom_right[y-1, x] == 255 
                            or bottom_right[y+1, x-1] == 255 
                            or bottom_right[y+1, x+1] == 255
                            or bottom_right[y-1, x-1] == 255 
                            or bottom_right[y-1, x+1] == 255 
                            or bottom_right[y, x - 1] == 255 
                            or bottom_right[y, x + 1] == 255):
                        bottom_right[y,x] = 255
                    else:
                        bottom_right[y,x] = 0
                except IndexError as e:
                    pass

    # combine all passes and prevent numbers from exceeding 255
    final_magnitude = top_left + top_right + bottom_left + bottom_right
    final_magnitude[final_magnitude > 0] = 255
 
    return top_left


# read in the image and the sigma value
# convert the image to greyscale
# copy the image
originalImage = cv2.imread(args["image"], 0)
sigma = args["sigma"]
image = originalImage.copy()

# assign threshhold values
low = args["low"]
high = args["high"]

# my implementation of Canny Edge Detection
output = edgeDetection(image, sigma, low, high)

cv2.imwrite("output.png", output)
