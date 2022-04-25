#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <chrono>


int main()
{

    //variables to hold various image states during the edge detection process
    cv::Mat imgOriginal; 
    cv::Mat imgCanny; 

    double sigma = 1.2;
    
    //read from the folder containing images
    std::string folder = "/media/marktrovinger/Datasets/seg_train/*.jpg";
    std::vector<cv::String> fn;
    cv::glob(folder, fn, false);

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> output_images;
    size_t count = fn.size();

    for (size_t i=0; i<count; i++)
        images.push_back(cv::imread(fn[i]));

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
    std::cout << "Detecting lines with a sigma value of: " << sigmaStr << std::endl;
    
    // start timing here
    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<count; i++){
        cv::Mat output_image;
        cv::Canny(images[i], output_image, 100, 200, 3);
        output_images.push_back(output_image);
    }
    auto stopTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = stopTime-startTime;
    std::cout << "Elapsed time for reference serial implementation (OpenCV): " << elapsed_seconds.count() << "s\n";

    //EdgeDetection(imgOriginal, imgCanny, 100, 200, sigma);

    //CV_WINDOW_AUTOSIZE will fix the window to image size
    //namedWindow("imgOriginal",CV_WINDOW_AUTOSIZE);        
    //namedWindow("imgCanny", CV_WINDOW_AUTOSIZE);
    
    for (size_t i=0; i<count; i++){
        cv::String filename = "output_";
        filename.append(std::to_string(i));
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

    //waitKey(0);
    return 0;
}