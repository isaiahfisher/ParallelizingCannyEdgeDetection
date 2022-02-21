#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat* edgeDetection(Mat image, float sigma, float low, float high){
    // Step 1: Gaussian
    // check to make sure we can open the image
    if(image.empty()){
        cout << "Could not find the image!" << endl;
        return -1;
    }

    Mat blurred_image;
    GaussianBlur(image, blurred_image, Size(5,5), 0);

    // Step 2: Gradient Calculation
    // Apply Sobel filter
    Mat xGradient, yGradient;
    Sobel(blurred_image, xGradient, CV_64F, 1, 0, ksize=3);
    Sobel(blurred_image, yGradient, CV_64F, 0, 1, ksize=3);

    // convert to polar coordinates
    Mat magnitute, angle;
    cartToPolar(xGradient, yGradient, magnitude, angle, angleInDegrees = true);
    
    // non-Maximum suppression
    // loop through all pixels in the image
    // according to https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv
    // using classic C pointers is the most efficient
    
    int channels = image.channels();
    int nRows = image.rows;
    int nCols = image.cols * channels();
    int firstNeighborX, firstNeighborY, secondNeighborX, secondNeighborY = 0;
    int i, j, currAngle;
    uchar* p;
    for(i = 0; i < nRows; ++i){
        p = image<ptr>char(i);
        for(j = 0; j < nCols; ++j){
            int currAngle = angle[p];
            if(currAngle > 180){
                currAngle -= 180;
            }
            if(currAngle <= 22.5 || currAngle > 157.5){
                firstNeighborX = p - 1;
                firstNeighborY = j;
                secondNeighborX = p + 1;
                secondNeighborY = j;
            }
            else if(currAngle > 22.5 && currAngle <= 67.5){
                firstNeighborX = p - 1;
                firstNeighborY = j - 1;
                secondNeighborX = p + 1;
                secondNeighborY = j;
            }
        }
    }
}

