/*
* CS 590 - High Performance Computing
* Term Project - Parallelizing Canny Edge Detection
* 
* Isiah Fisher, Mark Trovinger, Omer Yurdabakan
*
* canny_par.cpp - Edge Detection using MPI.
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#define LOW 100
#define HIGH 200
//#include "cvstd.hpp"

using namespace cv;
using namespace std;

/* 
*   Function for detecting edges in input images.
*
*   @param path - Path to input file
*   @param endwith - output image suffix
*   @param rank - used for MPI
*   @param size - used for MPI
*/
void EdgeDetection(string path, int endwith, int &rank, int &size)
{
    
    string img_addr = path;
    double sigma = 1.2;
    Mat    input = imread(img_addr);
    
    
    Mat output;
    //Mat input;
    Mat _magnitude;
    Mat magnitude;
    //uchar *_magnitude;
    int rows, cols;
    if (rank == 0)
    {
        // declare Mat fields that will hold different image manipulation stages
        Mat imgGrayscale, imgBlurred, xGradient, yGradient, angle;
        
        // get the grayscale image then blur it using a 5 x 5 gaussian kernal
        cvtColor(input, imgGrayscale, CV_BGR2GRAY);
        //printf("\n----%d---\n", imgGrayscale.depth());
        GaussianBlur(imgGrayscale, imgBlurred, Size(5, 5), sigma);
        // get the gradients of the image using Sobel kernal
        Sobel(imgBlurred, xGradient, CV_64F, 1, 0, 3);
        Sobel(imgBlurred, yGradient, CV_64F, 0, 1, 3);
        rows = imgBlurred.rows;
        cols = imgBlurred.cols;
        //bool check = imwrite("output.jpg",imgBlurred);
        // grab polar coordinates from the gradients
        cartToPolar(xGradient, yGradient, magnitude, angle, true);
        //printf("\n----%d---\n", angle.depth());
        //_magnitude.convertTo(magnitude, CV_32S);
        
        //printf("\n----%d---\n", magnitude.depth());
        
        // loop through all pixels in the image
        for (int i = 0; i < imgBlurred.cols; i++)
        {
            for (int j = 0; j < imgBlurred.rows; j++)
            {

                int currAngle = (int)angle.at<double>(j, i);
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
                }
                else if (currAngle > 22.5 && currAngle <= 67.5)
                {
                    firstNeighborX = i - 1;
                    firstNeighborY = j - 1;
                    secondNeighborX = i + 1;
                    secondNeighborY = j + 1;
                }
                else if (currAngle > 67.5 && currAngle <= 112.5)
                {
                    firstNeighborX = i - 1;
                    firstNeighborY = j + 1;
                    secondNeighborX = i;
                    secondNeighborY = j - 1;
                }
                else if (currAngle > 112.5 && currAngle <= 157.5)
                {
                    firstNeighborX = i;
                    firstNeighborY = j + 1;
                    secondNeighborX = i;
                    secondNeighborY = j - 1;
                }

                // check each pixel against its neighbor
                if (firstNeighborX >= 0 && firstNeighborY >= 0)
                {
                    if (input.cols > firstNeighborX && input.rows > firstNeighborY)
                    {
                        if (magnitude.at<double>(j, i) < magnitude.at<double>(firstNeighborY, firstNeighborX))
                        {
                            magnitude.at<double>(j, i) = 0;
                        }
                    }
                }
                if (secondNeighborX >= 0 && secondNeighborY >= 0)
                {
                    if (input.cols > secondNeighborX && input.rows > secondNeighborY)
                    {
                        if (magnitude.at<double>(j, i) < magnitude.at<double>(secondNeighborY, secondNeighborX))
                        {
                            magnitude.at<double>(j, i) = 0;
                        }
                    }
                }
            }
        }

        // Hysteresis thresholding
        // loop through all pixels in the image and discard 0s
        // 100 = weak threshold 255 = strong threshhold
        //_magnitude = new uchar[rows*cols];
        for (int i = 0; i < imgBlurred.cols; i++)
        {
            for (int j = 0; j < imgBlurred.rows; j++)
            {
                
                int currMagnitude = (int)magnitude.at<double>(j, i);
                
                if (currMagnitude < LOW)
                {
                    magnitude.at<double>(j, i) = 0;
                }
                else if (currMagnitude >= 100 && currMagnitude < HIGH)
                {
                    magnitude.at<double>(j, i) = 100;
                }
                else
                {
                    magnitude.at<double>(j, i) = 255;
                }
                //_magnitude[i*rows+j] = magnitude.at<uint64>(j, i);
            }
        }
        
    }
    
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    if(rank!=0) magnitude.create(rows,cols,CV_64F);
    MPI_Bcast(magnitude.data, rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    Mat image_clone;
    //Mat mag(rows,cols,CV_64F);
    //std::memcpy(mag.data, magnitude.data, cols*rows*sizeof(uint64));
    
    // start MPI
    
    // starting in top left determine if a weak threshold should be kept
    // or discarded
    if (rank == 0)
    {
        bool check = imwrite("test.jpg",magnitude);
        image_clone = magnitude.clone();
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if ((int)image_clone.at<uchar>(j, i) == 100)
                {
                    try
                    {
                        if ((int)image_clone.at<uchar>(j + 1, i) == 255 || (int)image_clone.at<uchar>(j - 1, i) == 255 || (int)image_clone.at<uchar>(j + 1, i - 1) == 255 || (int)image_clone.at<uchar>(j + 1, i + 1) == 255 || (int)image_clone.at<uchar>(j - 1, i - 1) == 255 || (int)image_clone.at<uchar>(j - 1, i + 1) == 255 || (int)image_clone.at<uchar>(j, i - 1) == 255 || (int)image_clone.at<uchar>(j, i + 1) == 255)
                        {
                            image_clone.at<uchar>(j, i) = 255;
                        }
                        else
                        {
                            image_clone.at<uchar>(j, i) = 0;
                        }
                    }
                    catch (...) // catches all exceptions
                    {
                        // equivalent to python pass I believe
                        ;
                    }
                }
            }
        }
        //printf("Here!\n");
    }
    if (rank == 1)
    {
        // starting from top right instead
        image_clone = magnitude.clone();
        for (int i = cols - 1; i > 0; i--)
        {
            for (int j = 0; j < rows - 1; j++)
            {
                if ((int)image_clone.at<uchar>(j, i) == 100)
                {
                    try
                    {
                        if ((int)image_clone.at<uchar>(j + 1, i) == 255 || (int)image_clone.at<uchar>(j - 1, i) == 255 || (int)image_clone.at<uchar>(j + 1, i - 1) == 255 || (int)image_clone.at<uchar>(j + 1, i + 1) == 255 || (int)image_clone.at<uchar>(j - 1, i - 1) == 255 || (int)image_clone.at<uchar>(j - 1, i + 1) == 255 || (int)image_clone.at<uchar>(j, i - 1) == 255 || (int)image_clone.at<uchar>(j, i + 1) == 255)
                        {
                            image_clone.at<uchar>(j, i) = 255;
                        }
                        else
                        {
                            image_clone.at<uchar>(j, i) = 0;
                        }
                    }
                    catch (...)
                    {
                        ;
                    }
                }
            }
        }
    }

    if (rank == 2)
    {
        // starting from bottom left instead
        image_clone = magnitude.clone();
        for (int i = 0; i < cols - 1; i++)
        {
            for (int j = rows - 1; j > 0; j--)
            {
                if ((int)image_clone.at<uchar>(j, i) == 100)
                {
                    try
                    {
                        if ((int)image_clone.at<uchar>(j + 1, i) == 255 || (int)image_clone.at<uchar>(j - 1, i) == 255 || (int)image_clone.at<uchar>(j + 1, i - 1) == 255 || (int)image_clone.at<uchar>(j + 1, i + 1) == 255 || (int)image_clone.at<uchar>(j - 1, i - 1) == 255 || (int)image_clone.at<uchar>(j - 1, i + 1) == 255 || (int)image_clone.at<uchar>(j, i - 1) == 255 || (int)image_clone.at<uchar>(j, i + 1) == 255)
                        {
                            image_clone.at<uchar>(j, i) = 255;
                        }
                        else
                        {
                            image_clone.at<uchar>(j, i) = 0;
                        }
                    }
                    catch (...)
                    {
                        ;
                    }
                }
            }
        }
    }

    if (rank == 3)
    {
        // starting from bottom right instead
        image_clone = magnitude.clone();
        for (int i = cols - 1; i > 0; i--)
        {
            for (int j = rows - 1; j > 0; j--)
            {
                if ((int)image_clone.at<uchar>(j, i) == 100)
                {
                    try
                    {
                        if ((int)image_clone.at<uchar>(j + 1, i) == 255 || (int)image_clone.at<uchar>(j - 1, i) == 255 || (int)image_clone.at<uchar>(j + 1, i - 1) == 255 || (int)image_clone.at<uchar>(j + 1, i + 1) == 255 || (int)image_clone.at<uchar>(j - 1, i - 1) == 255 || (int)image_clone.at<uchar>(j - 1, i + 1) == 255 || (int)image_clone.at<uchar>(j, i - 1) == 255 || (int)image_clone.at<uchar>(j, i + 1) == 255)
                        {
                            image_clone.at<uchar>(j, i) = 255;
                        }
                        else
                        {
                            image_clone.at<uchar>(j, i) = 0;
                        }
                    }
                    catch (...)
                    {
                        ;
                    }
                }
            }
        }
        //bool check = imwrite("output.jpg",image_clone);
    }

    // combine all versions and make all edges value 255
    Mat final_magnitude;
    if(rank==0){
        final_magnitude.create(rows, cols, CV_64F);
    }
    
    MPI_Reduce(image_clone.data, final_magnitude.data, cols * rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        //Mat final_magnitude_mat(rows, cols, CV_64F);
        //std::memcpy(final_magnitude_mat.data, final_magnitude, cols*rows*sizeof(double));
        //printf("\n----%d---\n", final_magnitude.depth());
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if ((int)final_magnitude.at<uchar>(j, i) > 0)
                {
                    final_magnitude.at<uchar>(j, i) = 255;
                }
            }
        }
        string name;
        name = path.substr(0,path.find('.'));
        stringstream ss;
        ss << name << endwith << ".jpg";
        cout<< ss.str()<<endl;
        bool check = imwrite(ss.str(), final_magnitude);
    }
    
    
}

int main()
{

    // variables to hold various image states during the edge detection process
    std::ifstream myfile("files.txt");
    struct timespec start, end;
    
    string line;
    int i = 0;
    int rank, size;
    MPI_Init(0, 0);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    if (rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &start);
    }
    if (myfile.is_open())
    {
         while (myfile>>line)
        { 
            EdgeDetection(line,i, rank, size);
            i++;
        }
    }
    if (rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
        time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
        std::cout << std::fixed
            << time_taken << std::setprecision(9) << " sec." << std::endl;
    }
    
   
    MPI_Finalize();
    return 0;
}
