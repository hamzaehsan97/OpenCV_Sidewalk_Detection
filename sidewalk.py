import cv2
import numpy as np
import math

# mask = cv2.inRange(gray, 55, 80)
# res = cv2.bitwise_and(gray, gray, mask=mask)


def returnImg(path):
    img = cv2.imread(path)
    return img

# return a gray image
def grayImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#Blur the image to reduce noise
#Return the cannyTransformed image with edges
def cannyImage(gray):
    blur = cv2.GaussianBlur(gray,(5,5),10)
    edges = cv2.Canny(blur, 50, 150)
    return edges

#Draw lines on the gray image with hough line transform
def drawLines(segment,img):
    lines = cv2.HoughLinesP(segment, 1, np.pi / 180, 400, minLineLength = 700, maxLineGap = 10)
    lines2 = calculate_lines(img, lines)
    lines_visualize = visualize_lines(img, lines2)
    cv2.imshow("hough", lines_visualize)
    output = cv2.addWeighted(img, 0.9, lines_visualize, 1, 1)
    cv2.imshow("output", output)
    # if lines.size == 0:
    #     print("No lines discovered under criteria")
    # else:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 5)

def returnSegment(img):
    #get image height
    height = img.shape[0]

    # create a triangle in the image for lanes
    polygons = np.array([[(0, height), (800, height), (380, 290)]])

    #create an image filled with zero intensities
    mask = np.zeros_like(img)

    #Allow the masked image to fill with 1's in the polygon
    cv2.fillPoly(mask, polygons, 255)

    #Create a triangular area for segmentation in the real img by over laying the mask on top
    segment = cv2.bitwise_and(img, mask)

    #return the segment
    return segment



def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize


def main():
    img = returnImg("/Users/hamzaehsan/Desktop/AI_Rescue_Drone/OpenCV_Sidewalk_Detection/images/AI_Rescue_Drone.jpeg")
    gray = grayImg(img)
    edges = cannyImage(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength = (img.shape[0])/2, maxLineGap = 100)
    for line in lines:
        (x1,y1,x2,y2)  = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 5)
    # segment= returnSegment(gray)
    # drawLines(segment,img)

    # cv2.imshow('mask',edges)
    cv2.imshow('img',img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()