#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace cv;
using namespace std;

class ImageContainer {
public:
  ImageContainer() { }
  ImageContainer(const Mat& _image, const Mat& _radMap, const vector<Point>& _pointVec) {
    _image.copyTo(this->image);
    _radMap.copyTo(this->radMap);
    this->pointVec.assign(_pointVec.begin(), _pointVec.end());
  }
  
  ImageContainer(const Mat& _image, const Mat& _radMap, 
                 const Mat& _Ix, const Mat& _Iy, const vector<Point>& _pointVec) {
    _image.copyTo(this->image);
    _radMap.copyTo(this->radMap);
    _Ix.copyTo(this->Ix);
    _Iy.copyTo(this->Iy);
    this->pointVec.assign(_pointVec.begin(), _pointVec.end());
  }
  
  Mat getImage() { return this->image; }
  Mat getRadMap() { return this->radMap; }
  Mat getIx() { return this->Ix; }
  Mat getIy() { return this->Iy; }
  vector<Point> getPoints() { return this->pointVec;}
   
  Mat getFeatureImage() {
    Mat newImage;
    this->image.copyTo(newImage);
    for (int i = 0; i < this->pointVec.size(); ++i) {
      Point point = pointVec[i];
      int x = point.x;
      int y = point.y;
      newImage.at<Vec3b>(x, y)[0] = 0;
      newImage.at<Vec3b>(x, y)[1] = 0;
      newImage.at<Vec3b>(x, y)[2] = 255;
    }
    return newImage; 
  }

  Mat getResponseMap() {
    double min, max;
    Mat normMap, cmImg;
    
    minMaxLoc(this->radMap, &min, &max);
    normalize(this->radMap, normMap, 1.0, 0.0, NORM_MINMAX);
    minMaxLoc(normMap, &min, &max);
    normMap.convertTo(normMap, CV_8U, 255, 0);
    minMaxLoc(normMap, &min, &max);
    applyColorMap(normMap, cmImg, COLORMAP_JET);
    
    return cmImg;
  }
  
  vector<Mat> getWindowsOfFeatures() {
    Mat DX, DY;
    vector<Mat> windows;
    double min, max;

    Ix.convertTo(DX, CV_32F, 1.0, 0);
    Iy.convertTo(DY, CV_32F, 1.0, 0); 
    minMaxLoc(Ix, &min, &max);
    cout << min << " " << max << endl;
    for (int i = 0; i < this->pointVec.size(); ++i) {
      Point p = this->pointVec[i];
      int x = p.x, y = p.y;
      float dx = DX.at<float>(x, y);
      float dy = DY.at<float>(x, y);
      float root = sqrt(dx*dx + dy*dy);
      float scale = 10.0;
      float newX = scale * dx / root;
      float newY = scale * dy / root;
      float vec[] = {newX, newY};
      Mat Vec(1, 2, CV_32FC1, vec);
      cout << x << " "  << y << " " << Vec << " " << fastAtan2(newY, newX) << endl;
      arrowedLine(this->image, Point(y, x), Point(int(newY+y), int(newX+x)), Scalar(255, 0, 0)); 
      //circle(this->image, Point(y, x), 10, Scalar(0, 255, 0));

      // normalize(Vec, Vec, 2.0, 0.0, NORM_MINMAX);
      // minMaxLoc(Vec, &min, &max);
      // cout << min << " " << max << endl;
    }
    
    return windows;
  }

  void getMatchEdges(ImageContainer *otherImage) {
    // Retreive all features from both imageA and imageB
    // Below is algorithm:
    // 1. Get a window of features points.
    // 2. Rotate the feature descriptor to up.
    // 3. Slice blocks of the images. 
    this->getWindowsOfFeatures(); 
    otherImage->getWindowsOfFeatures();
  }
  
private:
  Mat image;
  Mat radMap;
  Mat Ix, Iy;
  vector<Point>pointVec;
};

Mat getRadianceMap(const Mat& image, Mat& Ix, Mat& Iy, int winSize=1, float k=0.05) {
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
  // cout << blurImage.type() << " " << blurImage.channels()  << endl;
  cvtColor(blurImage, grayImage, CV_BGR2GRAY);
  
  // Sobel only support uchar operation
  grayImage.convertTo(grayImage, CV_16S, 255, 0) ; 
  Sobel(grayImage, DX,  -1, 1, 0, 5, scale, delta, BORDER_DEFAULT);
  Sobel(grayImage, DY,  -1, 0, 1, 5, scale, delta, BORDER_DEFAULT);
  Ix = DX.clone();
  Iy = DY.clone();
  convertScaleAbs(DY, DY);
  convertScaleAbs(DX, DX);
   
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
  normalize(radMap, radMap, 1.0, 0.0, NORM_MINMAX);
  minMaxLoc(radMap, &min, &max);
  cout << min << " " << max << endl; 
  float sumOfRadmap = sum(radMap)[0];
  float threshold = 1.6 * sumOfRadmap / radMap.rows / radMap.cols;

  for (int i = 1; i < radMap.rows - 1; ++i) {
    for (int j = 1; j < radMap.cols - 1; ++j) {
      if (radMap.at<float>(i, j) > threshold) {
        // If value in other pixel is lager than one in current pixel,
        // kill current pixel
        float wMin, wMax;
        int top = i - 1;
        int left = j - 1;
        int length = 3;
        bool isLarger = false;
        bool isLess = false;
        Mat window = radMap(Rect(left, top, length, length));
        for (int u = 0; u < length; ++u) {
          for (int v = 0; v < length; ++v) {
            if (u == 1 && v == 1)
              continue;
            if (window.at<float>(u, v) < window.at<float>(1, 1)) {
              isLarger = true;
              // break;
            }
            if (window.at<float>(u, v) >  window.at<float>(1, 1)) {
              isLess = true;
              // break;
            }
          }
        }
              
        // If current pixel is local minimum,
        // we should kill and delete this pixel.
        //if (!isLarger)
        //  KillMap.at<uchar>(i, j) = 1;

        if (!isLess)
          KillMap.at<uchar>(i, j) = 1;
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

void featureMatching(const vector<ImageContainer*>& images) {
  // Current only support two images
  ImageContainer *imageA = images[0];
  ImageContainer *imageB = images[1];
  
  imageA->getMatchEdges(imageB); 
}

int main( int argc, char** argv )
{

  if (argc < 4) {
    cout << "Usage: ./EXEC ${DATYA_DIRECTORY} ${IMAGE_PREFIX} ${NUM_OF_IMAGES}" << endl;
    return -1;
  }

  vector <ImageContainer*> images;
  String imageDir(argv[1]);
  String imagePrefix(argv[2]);
  int numOfImages = atoi(argv[3]);
  
  /*
    Create response map and feature points for each images
  */
  for (int idx = 0; idx < numOfImages; ++idx) {  
    String number(to_string(idx)); // by default
    if (number.length() == 1)
      number = "0" + number;

    String imagePath = imageDir + "/" + imagePrefix + number + ".jpg";
    cout << imagePath << endl;

    Mat inImage, image;
    inImage = imread(imagePath, IMREAD_COLOR); // Read the file
    if(inImage.empty()) {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
   
    Mat radMap, Ix, Iy;
    vector<Point> pointVec;
    inImage.convertTo(image, CV_32F, 1.0/255, 0) ; 
    radMap = getRadianceMap(image, Ix, Iy, 2, 0.04);
    pointVec = findFeaturePoints(radMap);
    ImageContainer *newImage = new ImageContainer(
      inImage, radMap, Ix, Iy, pointVec
    ); 
    images.push_back(newImage);
  }
  
  featureMatching(images);
  
  Mat outputMat;
  bool firstTime = true;
  for (int idx = 0; idx < numOfImages; ++idx) {
    ImageContainer *newImage = images[idx];
    Mat FeatureImage = newImage->getFeatureImage();
    Mat ResponseMap = newImage->getResponseMap();
    if (firstTime) {
      Mat matArray[] = {FeatureImage, ResponseMap};
      hconcat(matArray, 2, outputMat);
      firstTime = false;
    } else {
      hconcat(outputMat, FeatureImage, outputMat);
      hconcat(outputMat, ResponseMap, outputMat);
    }
  }


  imshow( "Display window", outputMat);
  waitKey(0); // Wait for a keystroke in the window
  
  return 0;
}
