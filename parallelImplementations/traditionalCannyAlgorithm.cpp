#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <mpi.h>

//#include "cvstd.hpp"

using namespace cv;
using namespace std;


//Our canny edge detection implementation
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

    //loop through all pixels in the image
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
    for (int i = 0; i < imgBlurred.cols; i++)
    {
        for (int j = 0; j < imgBlurred.rows; j++)
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

    Mat image_clone;
    int size, rank;
    MPI_Init(0,0);                 // start MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // get number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // get rank
    //starting in top left determine if a weak threshold should be kept
    //or discarded
    if (rank == 0)
    {
        image_clone = magnitude.clone();
        for (int i = 0; i < imgBlurred.cols; i++)
        {
            for (int j = 0; j < imgBlurred.rows; j++)
            {
                if (image_clone.at<uchar>(j,i) == 100)
                {
                    try
                    {
                        if (image_clone.at<uchar>(j+1, i) == 255
                                || image_clone.at<uchar>(j-1,i) == 255
                                || image_clone.at<uchar>(j+1, i-1) == 255
                                || image_clone.at<uchar>(j+1, i+1) == 255
                                || image_clone.at<uchar>(j-1, i-1) == 255
                                || image_clone.at<uchar>(j-1, i+1) == 255
                                || image_clone.at<uchar>(j, i-1) == 255
                                || image_clone.at<uchar>(j, i+1) == 255)
                        {
                            image_clone.at<uchar>(j,i) = 255;
                        } else
                        {
                           image_clone.at<uchar>(j,i) = 0;
                        }
                    } catch (...) //catches all exceptions
                    {
                        //equivalent to python pass I believe
                        ;
                    }
                }
            }
        }
    }
    if (rank == 1)
    {
        //starting from top right instead
        image_clone = magnitude.clone();
        for (int i = imgBlurred.cols - 1; i > 0; i--)
        {
            for (int j = 0; j < imgBlurred.rows - 1; j++)
            {
                if (image_clone.at<uchar>(j,i) == 100)
                {
                    try
                    {
                        if (image_clone.at<uchar>(j+1, i) == 255
                                || image_clone.at<uchar>(j-1,i) == 255
                                || image_clone.at<uchar>(j+1, i-1) == 255
                                || image_clone.at<uchar>(j+1, i+1) == 255
                                || image_clone.at<uchar>(j-1, i-1) == 255
                                || image_clone.at<uchar>(j-1, i+1) == 255
                                || image_clone.at<uchar>(j, i-1) == 255
                                || image_clone.at<uchar>(j, i+1) == 255)
                        {
                            image_clone.at<uchar>(j,i) = 255;
                        } else
                        {
                            image_clone.at<uchar>(j,i) = 0;
                        }
                    } catch (...)
                    {
                        ;
                    }
                }
            }
        }
    }

    if (rank == 2)
    {
        //starting from bottom left instead
        image_clone = magnitude.clone();
        for (int i = 0; i < imgBlurred.cols - 1; i++)
        {
            for (int j = imgBlurred.rows - 1; j > 0; j--)
            {
                if (image_clone.at<uchar>(j,i) == 100)
                {
                    try
                    {
                        if (image_clone.at<uchar>(j+1, i) == 255
                                || image_clone.at<uchar>(j-1,i) == 255
                                || image_clone.at<uchar>(j+1, i-1) == 255
                                || image_clone.at<uchar>(j+1, i+1) == 255
                                || image_clone.at<uchar>(j-1, i-1) == 255
                                || image_clone.at<uchar>(j-1, i+1) == 255
                                || image_clone.at<uchar>(j, i-1) == 255
                                || image_clone.at<uchar>(j, i+1) == 255)
                        {
                            image_clone.at<uchar>(j,i) = 255;
                        } else
                        {
                            image_clone.at<uchar>(j,i) = 0;
                        }
                    } catch (...)
                    {
                        ;
                    }
                }
            }
        }
    }

    if (rank==3)
    {
        //starting from bottom right instead
        image_clone = magnitude.clone();
        for (int i = imgBlurred.cols - 1; i > 0; i--)
        {
            for (int j = imgBlurred.rows - 1; j > 0; j--)
            {
                if (image_clone.at<uchar>(j,i) == 100)
                {
                    try
                    {
                        if ((int)image_clone.at<uchar>(j+1, i) == 255
                                ||(int)image_clone.at<uchar>(j-1,i) == 255
                                ||(int)image_clone.at<uchar>(j+1, i-1) == 255
                                ||(int)image_clone.at<uchar>(j+1, i+1) == 255
                                ||(int)image_clone.at<uchar>(j-1, i-1) == 255
                                ||(int)image_clone.at<uchar>(j-1, i+1) == 255
                                ||(int)image_clone.at<uchar>(j, i-1) == 255
                                ||(int)image_clone.at<uchar>(j, i+1) == 255)
                        {
                            image_clone.at<uchar>(j,i) = 255;
                        } else
                        {
                            image_clone.at<uchar>(j,i) = 0;
                        }
                    } catch (...)
                    {
                        ;
                    }
                }
            }
        }
    }

    //combine all versions and make all edges value 255
    uchar* final_magnitude[imgBlurred.rows*imgBlurred.cols*magnitude.channels()];
    uchar* image_clone_array = image_clone.data;
    MPI_Reduce(image_clone_array, final_magnitude, imgBlurred.cols*imgBlurred.rows, MPI_UNSIGNED_CHAR, MPI_SUM, 0, MPI_COMM_WORLD);
    
    Mat final_magnitude_mat(imgBlurred.rows, imgBlurred.cols, magnitude.channels(), final_magnitude);

    if (rank == 0)
    {
        for (int i = 0; i < imgBlurred.cols; i++)
        {
            for(int j = 0; j < imgBlurred.rows; j++)
            {
                if (final_magnitude_mat.at<uchar>(j,i) > 0)
                {
                    final_magnitude_mat.at<uchar>(j,i) = 255;
                }
            }
        }
        output = final_magnitude_mat.clone();
    }
    MPI_Finalize();

}

int main()
{

    //variables to hold various image states during the edge detection process
    Mat imgOriginal; 
    Mat imgCanny; 

    //Prompting user input
    //cout << "Please enter an image filename(string): ";
    //string img_addr;
    //cin >> img_addr;
    cout << "Please enter a sigma value(double): ";
    double sigma;
    cin >> sigma;
    //read from the folder containing images
    std::string folder = "/media/marktrovinger/Datasets/seg_train/*.jpg";
    vector<cv::String> fn;
    cv::glob(folder, fn, false);

    vector<Mat> images;
    vector<Mat> output_images;
    size_t count = fn.size();

    //let the user know their selection and open the image
    //cout << "Searching for " + img_addr << endl;
    //imgOriginal = imread(img_addr);

    for (size_t i=0; i<count; i++)
        images.push_back(imread(fn[i]));

    //make sure the image exists
    /*if (imgOriginal.empty()) 
    {
        cout << "error: image not read from file\n\n";
        return(0); 
    }
    */
    //let the user know that edge detection has begun
    char sigmaStr[10];
    sprintf(sigmaStr, "%fs", sigma);
    cout << "Detecting lines with a sigma value of: " << sigmaStr << endl;

    for (size_t i=0; i<count; i++){
        Mat output_image;
        EdgeDetection(images[i], output_image, 100, 200, sigma);
        output_images.push_back(output_image);
    }

    //EdgeDetection(imgOriginal, imgCanny, 100, 200, sigma);

    //CV_WINDOW_AUTOSIZE will fix the window to image size
    //namedWindow("imgOriginal",CV_WINDOW_AUTOSIZE);        
    //namedWindow("imgCanny", CV_WINDOW_AUTOSIZE);
    
    for (size_t i=0; i<count; i++){
        string filename = "output_";
        filename.append(to_string(i));
        filename.append(".jpg");
        imwrite(filename, output_images[i]);
    }
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
