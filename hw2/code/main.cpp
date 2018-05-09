#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

Mat getRadianceMap(const Mat& image, int winSize=1, float k=0.05) {
  Mat radMap, grayImage;
  cvtColor(image, grayImage, CV_BGR2GRAY);
  grayImage.copyTo(radMap);
  
  float a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
  Mat A(3, 3, CV_32FC1, a);

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

  Sobel(grayImage, DX,  -1, 1, 0, 5, scale, delta, BORDER_DEFAULT);
  convertScaleAbs(DX, DX);
  
  Sobel(grayImage, DY,  -1, 0, 1, 5, scale, delta, BORDER_DEFAULT);
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
      
      GaussianBlur(dxdx, dxdx, Size(3,3), 0, 0, BORDER_DEFAULT);
      GaussianBlur(dydy, dydy, Size(3,3), 0, 0, BORDER_DEFAULT);
      GaussianBlur(dxdy, dxdy, Size(3,3), 0, 0, BORDER_DEFAULT);

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

/*
  Type of element in radMap is float 
*/
vector<Point> findFeaturePoints(Mat &radMap) {
  double min, max, average;
  vector<Point> pointVec;
  Mat KillMap = Mat::zeros(radMap.rows, radMap.cols, CV_8UC1); 
   
  // minMaxLoc(radMap, &min, &max);
  float sumOfRadmap = sum(radMap)[0];
  float threshold = sumOfRadmap / radMap.rows / radMap.cols;

  for (int i = 1; i < radMap.rows - 1; ++i) {
    for (int j = 1; j < radMap.cols - 1; ++j) {
      if (radMap.at<float>(i, j) > threshold) {
        // If value in other pixel is lager than one in current pixel,
        // kill current pixel
        float wMin, wMax;
        int top = i - 1;
        int left = j - 1;
        int length = 3;
        // Mat dx = DX(Rect(left, top, length, length));
        bool isLarger = false;
        Mat window = radMap(Rect(left, top, length, length));
        for (int u = 0; u < length; ++u) {
          for (int v = 0; v < length; ++v) {
            if (u == 1 && v == 1) {
              continue;
            } 
            if (window.at<float>(u, v) < window.at<float>(1, 1)) {
              isLarger = true;
              break;
            }
          }
        }
              
        if (!isLarger) {
          // If current pixel is local minimum,
          // we should kill and delete this pixel.
          KillMap.at<uchar>(i, j) = 1;
        }
      }    
    }
  }
  
  // Extract feature points
  pointVec.clear();
  for (int i = 1; i < KillMap.rows; ++i)
    for (int j = 1; j < KillMap.cols; ++j)
      if (KillMap.at<uchar>(i, j) == 1)
        pointVec.push_back(Point(i, j));
  return pointVec;
}

int main( int argc, char** argv )
{
  // String imageName( "../../data/parrington/prtn03.jpg"); // by default
  String imageName( "../../data/grail/grail02.jpg"); // by default
  if( argc > 1)
  {
    imageName = argv[1];
  }
  
  Mat inImage, image;
  inImage = imread( imageName, IMREAD_COLOR ); // Read the file
  if(inImage.empty()) {
    cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  inImage.convertTo(image, CV_32F, 1.0/255, 0) ; 
 
  Mat radMap;
  vector<Point> pointVec;
  radMap = getRadianceMap(image, 2, 0.04);
  pointVec = findFeaturePoints(radMap);
  
  for (int i = 0; i < pointVec.size(); ++i) {
    int x = pointVec[i].x;
    int y = pointVec[i].y;
    inImage.at<Vec3b>(x, y)[2] = 255;
  }
   
  Mat outputMat, grayImage, cmImg;
  cvtColor(image, grayImage, CV_BGR2GRAY);
  
  double min, max, average;
  minMaxLoc(radMap, &min, &max);
  cout << "Min & Max = " << min << " " << max << endl;
  normalize(radMap, radMap, 1.0, 0.0, NORM_MINMAX);
  radMap.convertTo(radMap, CV_8U, 255, 0);
  minMaxLoc(radMap, &min, &max);
  cout << "Min & Max = " << min << " " << max << endl;
  
  applyColorMap(radMap, cmImg, COLORMAP_JET);

  radMap.convertTo(radMap, CV_32F, 1.0/255, 0); 
  Mat matArray[] = {cmImg, inImage};
  hconcat(matArray, 2, outputMat);

  // namedWindow( "Display window", CV_WINDOW_AUTOSIZE); 
  imshow( "Display window", outputMat );
  waitKey(0); // Wait for a keystroke in the window
  
  return 0;
}
