/*
* CS 590 - High Performance Computing
* Term Project - Parallelizing Canny Edge Detection
* 
* Isiah Fisher, Mark Trovinger, Omer Yurdabakan
*
*  cannyAlgorithmOpenMP.cpp - Experiments using algorithmic parallelism.
*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <omp.h>
#include <chrono>
//#include "cvstd.hpp"

using namespace cv;
using namespace std;


/* 
*   Function for detecting edges in input images.
*
*   @param input - Input image in cv::Mat format
*   @param low - low threshold for hysterisis
*   @param high - high threshold for same
*   @param sigma - sigma value for Gaussian Blur
*   @param output - image after edge detection, cv::Mat format
*/
void EdgeDetection(Mat input, Mat &output, int low, int high, int sigma)
{
    //declare Mat fields that will hold different image manipulation stages
    Mat imgGrayscale, imgBlurred, xGradient, yGradient, magnitude, angle;
    
    //get the grayscale image then blur it using a 5 x 5 gaussian kernal
    cvtColor(input, imgGrayscale,CV_BGR2GRAY);
    GaussianBlur(imgGrayscale, imgBlurred, Size(5, 5), sigma); 
    //get the gradients of the image using Sobel kernal
    Sobel(imgBlurred, xGradient, CV_64F, 1, 0, 3);
    Sobel(imgBlurred, yGradient, CV_64F, 0, 1, 3);

    //grab polar coordinates from the gradients
    cartToPolar(xGradient, yGradient, magnitude, angle, true);

    int numThreads = 0;

    //loop through all pixels in the image
    #pragma omp parallel for
    for (int i = 0; i < imgBlurred.cols; i++)
    {
        for (int j = 0; j < imgBlurred.rows; j++)
        {

            uchar currAngle = angle.at<uchar>(j, i);
            int firstNeighborX;
            int firstNeighborY;
            int secondNeighborX;
            int secondNeighborY;
            if (currAngle > 180)
            {
                currAngle -= 180;
            }
            if (currAngle <= 22.5 || currAngle > 157.5)
            {
               firstNeighborX = i - 1;
               firstNeighborY = j;
            } else if (currAngle > 22.5 && currAngle <= 67.5)
            {
                firstNeighborX = i - 1;
                firstNeighborY = j - 1;
                secondNeighborX = i + 1;
                secondNeighborY = j + 1;
            } else if (currAngle > 67.5 && currAngle <= 112.5)
            {
                firstNeighborX = i - 1;
                firstNeighborY = j + 1;
                secondNeighborX = i;
                secondNeighborY = j - 1;
            } else if (currAngle > 112.5 && currAngle <= 157.5)
            {
                firstNeighborX = i;
                firstNeighborY = j + 1;
                secondNeighborX = i;
                secondNeighborY = j - 1;
            }
            
            //check each pixel against its neighbor
            if (firstNeighborX >= 0 && firstNeighborY >= 0)
            {
                if (input.cols > firstNeighborX && input.rows > firstNeighborY )
                {
                    if (magnitude.at<uchar>(j,i) < magnitude.at<uchar>(firstNeighborY, firstNeighborX))
                    {
                        magnitude.at<uchar>(j,i) = 0;
                    }
                }
            }
            if (secondNeighborX >= 0 && secondNeighborY >= 0)
            {
                if (input.cols > secondNeighborX && input.rows > secondNeighborY )
                {
                    if (magnitude.at<uchar>(j,i) < magnitude.at<uchar>(secondNeighborY, secondNeighborX))
                    {
                        magnitude.at<uchar>(j,i) = 0;
                    }
                }
            }
        }
    }

    //Hysteresis thresholding
    //loop through all pixels in the image and discard 0s
    //100 = weak threshold 255 = strong threshhold
    #pragma omp parallel 
    {
        int j = 0;
        numThreads =  omp_get_num_threads();
        #pragma omp for 
        for (int i = 0; i < imgBlurred.cols; i++)
        {
            for (j = 0; j < imgBlurred.rows; j++)
            {
                uchar currMagnitude = magnitude.at<uchar>(j,i);

                if (currMagnitude < low)
                {
                    magnitude.at<uchar>(j,i) = 0;
                } else if (currMagnitude >=100 && currMagnitude < high)
                {
                    magnitude.at<uchar>(j,i) = 100;
                } else
                {
                    magnitude.at<uchar>(j,i) = 255;
                }
            }
        }
    }

    //std::cout << "Using " << numThreads << " threads!" << std::endl;

    //starting in top left determine if a weak threshold should be kept
    //or discarded
    Mat top_left = magnitude.clone();
    for (int i = 0; i < imgBlurred.cols; i++)
    {
        for (int j = 0; j < imgBlurred.rows; j++)
        {
            if (top_left.at<uchar>(j,i) == 100)
            {
                try
                {
                    if (top_left.at<uchar>(j+1, i) == 255
                            || top_left.at<uchar>(j-1,i) == 255
                            || top_left.at<uchar>(j+1, i-1) == 255
                            || top_left.at<uchar>(j+1, i+1) == 255
                            || top_left.at<uchar>(j-1, i-1) == 255
                            || top_left.at<uchar>(j-1, i+1) == 255
                            || top_left.at<uchar>(j, i-1) == 255
                            || top_left.at<uchar>(j, i+1) == 255)
                    {
                        top_left.at<uchar>(j,i) = 255;
                    } else
                    {
                        top_left.at<uchar>(j,i) = 0;
                    }
                } catch (...) //catches all exceptions
                {
                    //equivalent to python pass I believe
                    ;
                }
            }
        }
    }

    //starting from top right instead
    Mat top_right = magnitude.clone();
    for (int i = imgBlurred.cols - 1; i > 0; i--)
    {
        for (int j = 0; j < imgBlurred.rows - 1; j++)
        {
            if (top_right.at<uchar>(j,i) == 100)
            {
                try
                {
                    if (top_right.at<uchar>(j+1, i) == 255
                            || top_right.at<uchar>(j-1,i) == 255
                            || top_right.at<uchar>(j+1, i-1) == 255
                            || top_right.at<uchar>(j+1, i+1) == 255
                            || top_right.at<uchar>(j-1, i-1) == 255
                            || top_right.at<uchar>(j-1, i+1) == 255
                            || top_right.at<uchar>(j, i-1) == 255
                            || top_right.at<uchar>(j, i+1) == 255)
                    {
                        top_right.at<uchar>(j,i) = 255;
                    } else
                    {
                        top_right.at<uchar>(j,i) = 0;
                    }
                } catch (...)
                {
                    ;
                }
            }
        }
    }

    //starting from bottom left instead
    Mat bottom_left = magnitude.clone();
    for (int i = 0; i < imgBlurred.cols - 1; i++)
    {
        for (int j = imgBlurred.rows - 1; j > 0; j--)
        {
            if (bottom_left.at<uchar>(j,i) == 100)
            {
                try
                {
                    if (bottom_left.at<uchar>(j+1, i) == 255
                            || bottom_left.at<uchar>(j-1,i) == 255
                            || bottom_left.at<uchar>(j+1, i-1) == 255
                            || bottom_left.at<uchar>(j+1, i+1) == 255
                            || bottom_left.at<uchar>(j-1, i-1) == 255
                            || bottom_left.at<uchar>(j-1, i+1) == 255
                            || bottom_left.at<uchar>(j, i-1) == 255
                            || bottom_left.at<uchar>(j, i+1) == 255)
                    {
                        bottom_left.at<uchar>(j,i) = 255;
                    } else
                    {
                        bottom_left.at<uchar>(j,i) = 0;
                    }
                } catch (...)
                {
                    ;
                }
            }
        }
    }

    //starting from bottom right instead
    Mat bottom_right = magnitude.clone();
    for (int i = imgBlurred.cols - 1; i > 0; i--)
    {
        for (int j = imgBlurred.rows - 1; j > 0; j--)
        {
            if (bottom_right.at<uchar>(j,i) == 100)
            {
                try
                {
                    if ((int)bottom_right.at<uchar>(j+1, i) == 255
                            ||(int)bottom_right.at<uchar>(j-1,i) == 255
                            ||(int)bottom_right.at<uchar>(j+1, i-1) == 255
                            ||(int)bottom_right.at<uchar>(j+1, i+1) == 255
                            ||(int)bottom_right.at<uchar>(j-1, i-1) == 255
                            ||(int)bottom_right.at<uchar>(j-1, i+1) == 255
                            ||(int)bottom_right.at<uchar>(j, i-1) == 255
                            ||(int)bottom_right.at<uchar>(j, i+1) == 255)
                    {
                        bottom_right.at<uchar>(j,i) = 255;
                    } else
                    {
                        bottom_right.at<uchar>(j,i) = 0;
                    }
                } catch (...)
                {
                    ;
                }
            }
        }
    }

    //combine all versions and make all edges value 255
    Mat final_magnitude = top_left + top_right + bottom_left + bottom_right;
    for (int i = 0; i < imgBlurred.cols; i++)
    {
        for(int j = 0; j < imgBlurred.rows; j++)
        {
            if (final_magnitude.at<uchar>(j,i) > 0)
            {
               final_magnitude.at<uchar>(j,i) = 255;
            }
        }
    }
    output = final_magnitude.clone();

}

int main()
{

    //variables to hold various image states during the edge detection process
    Mat imgOriginal; 
    Mat imgCanny; 
    double sigma = 1.2;
    //std::string folder = "/media/marktrovinger/Datasets/seg_train/*.jpg";
    std::string ScholarFolder = "/home/mtroving/HPC/images";
    vector<cv::String> fn;
    cv::glob(ScholarFolder, fn, false);

    vector<Mat> images;
    vector<Mat> output_images;
    size_t count = fn.size();
    size_t testing_count = 300;

    for (size_t i=0; i<count; i++)
        images.push_back(imread(fn[i]));

    //let the user know that edge detection has begun
    char sigmaStr[10];
    sprintf(sigmaStr, "%fs", sigma);
    cout << "Detecting lines with a sigma value of: " << sigmaStr << endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<count; i++){
        Mat output_image;
        EdgeDetection(images[i], output_image, 100, 200, sigma);
        output_images.push_back(output_image);
    }
    auto stopTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = stopTime-startTime;
    std::cout << "Elapsed time for our OpenMP implementation: " << elapsed_seconds.count() << "s\n";
    std::cout << "Testing on " << count << " 4k images." << std::endl;

    //EdgeDetection(imgOriginal, imgCanny, 100, 200, sigma);

    //CV_WINDOW_AUTOSIZE will fix the window to image size
    //namedWindow("imgOriginal",CV_WINDOW_AUTOSIZE);        
    //namedWindow("imgCanny", CV_WINDOW_AUTOSIZE);
    
    /*
    for (size_t i=0; i<count; i++){
        cv::String filename = "output_";
        filename.append(to_string(i));
        filename.append(".jpg");
        imwrite(filename, output_images[i]);
    }
    */
    //Show windows
    //imshow("imgOriginal", imgOriginal);
    //bool check = imwrite("output.jpg", imgCanny);
    /*if (check == false) {
        cout << "Mission - Saving the image, FAILED" << endl;
  
        // wait for any key to be pressed
        cin.get();
        return -1;
    }  
    */
    //imshow("imgCanny", imgCanny);

    waitKey(0);
    return 0;
}
