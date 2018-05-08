#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Mat getRadianceMap(const Mat& image, int winSize=1, float k=0.05) {
  Mat radMap, grayImage;
  cvtColor(image, grayImage, CV_BGR2GRAY);
  grayImage.copyTo(radMap);
  
  float a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
  Mat A(3, 3, CV_32FC1, a);
  Mat Sob, Mul;
  Scalar summary = sum(A);
  cout << "sum = " << summary[0] << endl;
  A.at<float>(0, 0) = 100;
  cout << A << endl;

  /*
    Create dx, dy of images by using gaussian blur to reduce noise,
    and exploit convolution to calculate dx and dy.  
  */
  const int scale = 1;
  const int delta = 0;
  
  Mat blurImage, DX, DY, borderImage;
  GaussianBlur(image, blurImage, Size(3,3), 0, 0, BORDER_DEFAULT);
  cout << blurImage.type() << " " << blurImage.channels()  << endl;
  cvtColor(blurImage, grayImage, CV_BGR2GRAY);
  
  // Sobel only support uchar operation
  grayImage.convertTo(grayImage, CV_8U, 255, 0) ; 

  Sobel(grayImage, DX,  -1, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(DX, DX);
  
  Sobel(grayImage, DY,  -1, 0, 1, 3, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(DY, DY);
   
  // Calculate border image, just for debug
  addWeighted(DX, 0.5, DY, 0.5, 0, borderImage);
  
  // Convert DX DY back to float data type
  DX.convertTo(DX, CV_32F, 1.0/255, 0);
  DY.convertTo(DY, CV_32F, 1.0/255, 0);
  radMap = Mat::zeros(image.rows, image.cols, CV_32FC1); 
  for (int i = winSize; i < image.rows - winSize; ++i) {
    for (int j = winSize; j < image.cols - winSize; ++j) {
      
      // Calculate the window position and size in images
      int length = winSize * 2 + 1;
      int top = i - winSize;
      int left = j - winSize;
      
      // Calculate DX and DY in current windows
      Mat dxdx, dydy, dxdy;
      Mat dx = DX(Rect(left, top, length, length));
      Mat dy = DY(Rect(left, top, length, length));
      multiply(dx, dx, dxdx);
      multiply(dy, dy, dydy);
      multiply(dx, dy, dxdy);

      // Sum of all DX^2, DY^2, DXDY in current window
      float sumDx = sum(dxdx)[0];
      float sumDy = sum(dydy)[0];
      float sumDxDy = sum(dxdy)[0];
      float mArray[2][2] = {{sumDx, sumDxDy}, {sumDxDy, sumDy}};       
      
      // Calculate eigen value of matrix
      // Transform quadratic function of M to
      // symmetric matrix by (M + MT)/2 
      // Besides, find eigenvalues of symmetric matrix 'Symm' from M
      Mat M(2, 2, CV_32FC1, mArray);
      Mat MT, Sum, Mean(2, 2, CV_32FC1,  Scalar(2)), Symm;
      transpose(M, MT);
      add(M, MT, Sum);
      divide(Sum, Mean, Symm);
      Mat eigenValues, eigenVectors;
      eigen(Symm, eigenValues, eigenVectors);
      
      // After getting eigenvalues, we can calculate R
      float lambda1 = eigenValues.at<float>(0, 0);
      float lambda2 = eigenValues.at<float>(1, 0);
      float detM = lambda1 * lambda2;
      float traceM = lambda1 + lambda2;
      float R = detM - k * traceM * traceM; 
      radMap.at<float>(i, j) = R;
    }
  }
   
  // borderImage.convertTo(borderImage, CV_32F, 1.0/255, 0);
  // borderImage.copyTo(radMap);
  return radMap;
}

int main( int argc, char** argv )
{
  String imageName( "../../data/parrington/prtn01.jpg"); // by default
  if( argc > 1)
  {
    imageName = argv[1];
  }
  
  Mat inImage, image;
  inImage = imread( imageName, IMREAD_COLOR ); // Read the file
  inImage.convertTo(image, CV_32F, 1.0/255, 0) ; 
 
  Mat radMap;
  radMap = getRadianceMap(image, 1);
   
  cout << inImage.at<Vec3b>(0,0) << endl;
  cout << image.at<Vec3f>(0,0) << endl;
   
  if( radMap.empty() )                      // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
  
  Mat outputMat, grayImage, cmImg;
  cvtColor(image, grayImage, CV_BGR2GRAY);
  
  double min, max;
  minMaxLoc(radMap, &min, &max);
  cout << "Min & Max = " << min << " " << max << endl;
  normalize(radMap, radMap, 1.0, 0.0, NORM_MINMAX);
  radMap.convertTo(radMap, CV_8U, 255, 0);
  minMaxLoc(radMap, &min, &max);
  cout << "Min & Max = " << min << " " << max << endl;
  
  applyColorMap(radMap, cmImg, COLORMAP_JET);
  cout << cmImg.channels() << " " << cmImg.type() << " " << cmImg.rows << " " << cmImg.cols << endl;
  cout << image.channels() << " " << image.type() << " " << image.rows << " " << image.cols << endl;
  
  
  Mat matArray[] = {cmImg, inImage};
  hconcat(matArray, 2, outputMat);

  // namedWindow( "Display window", CV_WINDOW_AUTOSIZE); 
  imshow( "Display window", outputMat );                // Show our image inside it.
  waitKey(0); // Wait for a keystroke in the window
  
  return 0;
}
