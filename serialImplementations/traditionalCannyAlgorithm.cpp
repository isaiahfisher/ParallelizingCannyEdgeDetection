#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int edgeDetection(Mat image, float sigma, float low, float high){
    // Step 1: Gaussian
    // check to make sure we can open the image
    Mat detectedEdges;
    if(image.empty()){
        cout << "Could not find the image!" << endl;
        return -1;
    }

    Mat blurred_image;
    GaussianBlur(image, blurred_image, Size(5,5), 0);

    // Step 2: Gradient Calculation
    // Apply Sobel filter
    Mat xGradient, yGradient;
    Sobel(blurred_image, xGradient, CV_64F, 1, 0, 3);
    Sobel(blurred_image, yGradient, CV_64F, 0, 1, 3);

    // convert to polar coordinates
    Mat magnitude, angle;
    cartToPolar(xGradient, yGradient, magnitude, angle, true);
    cout << magnitude << endl;
    
    // non-Maximum suppression
    // loop through all pixels in the image
    // according to https://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv
    // using classic C pointers is the most efficient
    
    int channels = image.channels();
    int nRows = image.rows;
    int nCols = image.cols * channels;
    int firstNeighborX, firstNeighborY, secondNeighborX, secondNeighborY = 0;
    int i, j, currAngle;
    uchar* p;
    /*
    for(i = 0; i < nRows; ++i){
        p = image.ptr<uchar>(i);
        for(j = 0; j < nCols; ++j){
            int currAngle = angle[j][i];
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
            else if(currAngle > 67.5 && currAngle <= 112.5){
                firstNeighborX = p - 1;
                firstNeighborY = j + 1;
                secondNeighborX = p + 1;
                secondNeighborY = j - 1;
            }
            else if(currAngle > 112.5 && currAngle <= 157.5){
                firstNeighborX = p;
                firstNeighborY = j + 1;
                secondNeighborX = p;
                secondNeighborY = y - 1;
            }

            if(firstNeighborX >= 0 && firstNeighborY >= 0){
                if(nCols > firstNeighborX && nRows > firstNeighborY){
                    if(magnitude[j, p] < magnitude[firstNeighborX, firstNeighborY]){
                        magnitude[j, p] = 0;
                    }
                    }
            }
        }
        
    } */
    return 0;
}

int main(int argc, char* argv[]){
    // read image from disk, using cmd line
    // parameters, using tutorial parameters for now
    string image_path = samples::findFile("starry_night.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    edgeDetection(img, 0.0, 0.0, 0.0);
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return 0;
    // call Canny's algorithm, and output the 
    // edged image to disk

    
}
