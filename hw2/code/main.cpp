#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace cv;
using namespace std;

#define LENGTH 27
#define BLOCK_NUM 9
#define ERROR_RATE 1

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
    this->patches = this->getWindowsOfFeatures();
  }
  
  Mat getImage() { return this->image; }
  Mat getRadMap() { return this->radMap; }
  Mat getIx() { return this->Ix; }
  Mat getIy() { return this->Iy; }
  vector<Point> getPoints() { return this->pointVec;}
  vector<Mat> getPatches() { return this->patches; } 

  Mat getFeatureImage() {
    Mat newImage;
    this->image.copyTo(newImage);
    for (int i = 0; i < this->pointVec.size(); ++i) {
      Point point = pointVec[i];
      int x = point.x;
      int y = point.y;
      circle(newImage, Point(y, x), 5, Scalar(0, 0, 255));
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
  
  vector<Mat> getWindowsOfFeatures(int length=LENGTH, int blockNum=BLOCK_NUM) {
    Mat DX, DY;
    vector<Mat> windows;
    double min, max;

    Ix.convertTo(DX, CV_32F, 1.0, 0);
    Iy.convertTo(DY, CV_32F, 1.0, 0); 
    minMaxLoc(Ix, &min, &max);
    cout << min << " " << max << endl;
    for (int i = 0; i < this->pointVec.size(); ++i) {
      
      //  Compute vector rotation angle
      Point p = this->pointVec[i];
      int x = p.x, y = p.y;
      float dx = DX.at<float>(x, y);
      float dy = DY.at<float>(x, y);
      float root = sqrt(dx*dx + dy*dy);
      float scale = 10.0;
      float newX = scale * dx / root;
      float newY = scale * dy / root;
      float theta = fastAtan2(newY, newX);
      float pi = std::acos(-1);
      float angle = pi * theta / 180;
      // arrowedLine(this->image, Point(y, x), Point(int(newY+y), int(newX+x)), Scalar(255, 0, 0)); 
      
      // Transform and fill value in pixel to window 
      Mat window = Mat::zeros(length, length, CV_8UC3);
      int offset = (length - 1) / 2;
      for (int u = -offset; u <= offset; ++u)
        for (int v = -offset; v <= offset; ++v) {
          // Rotate and shift the point to original image
          int origX = (int)(u * cos(angle) - v * sin(angle) + 0.5) + x;
          int origY = (int)(u * sin(angle) + v * cos(angle) + 0.5) + y;
          int matX = u + offset;
          int matY = v + offset;
          // Map current window position to origin image
          window.at<Vec3b>(matX, matY) = this->image.at<Vec3b>(origX, origY);
        }
      
      Mat patch = Mat::zeros(blockNum, blockNum, CV_8UC3);
      int blockLength = length / blockNum;
      for (int u = 0; u < blockNum; ++u)
        for (int v = 0; v < blockNum; ++v) {
          int xx = u * blockLength;
          int yy = v * blockLength;

          // Maybe here shold be modified for capture the features
          Mat block = window(Rect(yy, xx, blockLength, blockLength));
          int numOfElemts = blockLength * blockLength;
          uchar B = sum(block)[0] / numOfElemts;
          uchar G = sum(block)[1] / numOfElemts;
          uchar R = sum(block)[2] / numOfElemts;
          patch.at<Vec3b>(u, v) = Vec3b(B, G, R);
        }
      windows.push_back(patch);
    }
    
    return windows;
  }
  
  string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	  case CV_8U:  r = "8U"; break;
	  case CV_8S:  r = "8S"; break;
	  case CV_16U: r = "16U"; break;
	  case CV_16S: r = "16S"; break;
	  case CV_32S: r = "32S"; break;
	  case CV_32F: r = "32F"; break;
	  case CV_64F: r = "64F"; break;
	  default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
  }

  void getMatchEdges(ImageContainer *otherImage) {
    // Retreive all features from both imageA and imageB
    // Below is algorithm:
    // 1. Get a window of features points.
    // 2. Rotate the feature descriptor to up.
    // 3. Slice blocks of the images.
    // 4. Use kdtree to find the matching of patches
    int length = LENGTH;
    int blockNum = BLOCK_NUM;
    int totalBlock = blockNum * blockNum;
    vector <Mat> patchA = this->getPatches();
    vector <Mat> patchB = otherImage->getPatches();
    vector <Point> pointA = this->getPoints();
    vector <Point> pointB = otherImage->getPoints();
  /* 
    Mat AA, BB, output;
    while(patchA.size() != patchB.size()) {
      if (patchA.size() > patchB.size())
        patchB.push_back(patchA[0]);
      else
        patchA.push_back(patchB[0]);
    }
    hconcat(patchA.data(), patchA.size(), AA);
    hconcat(patchB.data(), patchB.size(), BB);
    
    vconcat(AA, BB, output);
    imshow("hahah", output);
    waitKey(0);
  */
    
    // A lambda function for patches convertion
    auto trainDataCvt = [](const vector<Mat>& patchList, 
                           int totalBlock) -> pair<Mat, Mat> {
      Mat Data(patchList.size(), totalBlock, CV_32FC1);
      Mat Labels(patchList.size(), 1, CV_32SC1);
      for (int idx = 0; idx < patchList.size(); ++idx) {
        Mat patch;
        cvtColor(patchList[idx], patch, CV_BGR2GRAY);
        Labels.at<int>(idx, 0) = idx;
        // Copy all features to train data matrix
        for (int i = 0; i < patch.rows; ++i)
          for (int j = 0; j < patch.cols; ++j) {
            int curPos = i * patch.rows + j;
            Data.at<float>(idx, curPos) = patch.at<uchar>(i, j);
          }
      }
      pair<Mat, Mat> dataSet(Data, Labels);
      return dataSet;
    };
    
    auto matchFeatures = [](const pair<Mat, Mat>& A, const pair<Mat, Mat>& B) -> map<int, int> {
      Mat matchB2A, matchA2B;
      map <int, int> matchTable, B2ATable, A2BTable;
      int knnLevel = 2;
      
      Ptr<ml::KNearest> knn = ml::KNearest::create();
      knn->train(A.first, ml::ROW_SAMPLE, A.second);
      knn->findNearest(B.first, knnLevel, matchB2A); 
      for (int i = 0; i < matchB2A.rows; ++i) {
        // Create Label B to Label A Table
        int labelB = i;
        int labelA = matchB2A.at<float>(i, 0);
        B2ATable[labelB] = labelA; 
      }

      knn = ml::KNearest::create();
      knn->train(B.first, ml::ROW_SAMPLE, B.second);
      knn->findNearest(A.first, knnLevel, matchA2B); 
      for (int i = 0; i < matchA2B.rows; ++i) {
        // Create Label A to Label B table
        int labelA = i;
        int labelB = matchA2B.at<float>(i, 0);
        A2BTable[labelA] = labelB;
      }
      
      map<int, int>::iterator it;
      for (it = A2BTable.begin(); it != A2BTable.end(); ++it) {
        int labelA = it->first;
        int labelB = it->second;
        if (B2ATable[labelB] == labelA)
          matchTable[labelB] = labelA;
      }
      
      return matchTable;
    };
     
    pair <Mat, Mat> ASet, BSet;
    ASet = trainDataCvt(patchA, totalBlock);
    BSet = trainDataCvt(patchB, totalBlock);
    
    /*
      table format = [Label B] --> Label A
    */ 
    map <int, int> matchTable = matchFeatures(ASet, BSet); 
    map <int, int>::iterator it;
    
    Mat imageA = this->getFeatureImage();
    Mat imageB = otherImage->getFeatureImage();
    Mat matArray[] = { imageA, imageB };
    Mat outputMat;
    hconcat(matArray, 2, outputMat);
    
    for (it = matchTable.begin(); it != matchTable.end(); ++it) {
      int labelB = it->first;
      int labelA = it->second;
      
      // cout << pointA[labelA] << " " << pointB[labelB] << endl; 
      int Ax = pointA[labelA].x, Ay = pointA[labelA].y;
      int Bx = pointB[labelB].x, By = pointB[labelB].y + imageA.cols;
      
      Mat MatA, MatB, MatDiff;
      cvtColor(patchA[labelA], MatA, CV_BGR2GRAY);
      cvtColor(patchB[labelB], MatB, CV_BGR2GRAY);
      MatA.convertTo(MatA, CV_32F, 1.0/255, 0);
      MatB.convertTo(MatB, CV_32F, 1.0/255, 0); 

      subtract(MatA, MatB, MatDiff);
      multiply(MatDiff, MatDiff, MatDiff);
      float error = sum(MatDiff)[0]; 
      cout << "error rate = " << error << endl; 
      
      if (error < ERROR_RATE) 
        line(outputMat, Point(Ay, Ax), Point(By, Bx), Scalar(0, 255, 0)); 
    }
    // Mat trainData = dataSet.first;
    // Mat trainLabels = dataSet.second;
    // Mat testData = dataSet.first;
    // Mat testLabels = dataSet.second;
/*
    Mat bestLabels;
    Ptr<ml::KNearest> knn = ml::KNearest::create();
    knn->train(trainData, ml::ROW_SAMPLE, trainLabels);
    knn->findNearest(testData, 2, bestLabels); 
*/   
/*     
    cout << "BEST LABELS: " << patchA.size() << " " << this->type2str(bestLabels.type()) << " " << bestLabels.rows << " " << patchB.size() << " " << testLabels.rows << endl;
    for (int i = 0; i < patchB.size(); ++i) {
      int labelIdx = bestLabels.at<float>(i, 0);
      cout << "LABEL: " << labelIdx << endl;
      int Ax = pointA[labelIdx].x, Ay = pointA[labelIdx].y;
      int Bx = pointB[i].x, By = pointB[i].y + imageB.cols;
      line(outputMat, Point(Ay, Ax), Point(By, Bx), Scalar(0, 255, 0)); 
    }
*/    
    imshow("Window", outputMat);
    waitKey(0);
  }
  
private:
  Mat image;
  Mat radMap;
  Mat Ix, Iy;
  vector<Point> pointVec;
  vector<Mat> patches;
};

Mat getRadianceMap(const Mat& image, Mat& Ix, Mat& Iy, int winSize=1, float k=0.05) {
  Mat radMap, grayImage;
  cvtColor(image, grayImage, CV_BGR2GRAY);
  grayImage.copyTo(radMap);
 
  uchar a[3][3] = {{128,253,134},{4,5,6},{7,8,9}};
  Mat A(3, 3, CV_8UC1, a);
  // cout << "SUM: " << sum(A)[0] << endl; 
  
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
  Sobel(grayImage, DX,  -1, 1, 0, 3, scale, delta, BORDER_DEFAULT);
  Sobel(grayImage, DY,  -1, 0, 1, 3, scale, delta, BORDER_DEFAULT);
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
   
  return radMap;
}

/*
  Type of element in radMap is float 
*/
vector<Point> findFeaturePoints(Mat &radMap) {
  double min, max, average;
  vector<Point> pointVec;
  // Current KillMap is used to record local maximum
  Mat KillMap = Mat::zeros(radMap.rows, radMap.cols, CV_8UC1); 
  normalize(radMap, radMap, 1.0, 0.0, NORM_MINMAX);
  minMaxLoc(radMap, &min, &max);
  cout << min << " " << max << endl; 
  float sumOfRadmap = sum(radMap)[0];
  float threshold = 1.2 * sumOfRadmap / radMap.rows / radMap.cols;

  for (int i = 1; i < radMap.rows - 1; ++i) {
    for (int j = 1; j < radMap.cols - 1; ++j) {
      if (radMap.at<float>(i, j) > threshold) {
        // Find local maximum in the middle of window
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
            
            if (window.at<float>(u, v) >  window.at<float>(1, 1)) 
              isLess = true;
          }
        }
        
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
  
  random_shuffle(pointVec.begin(), pointVec.end()); 
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

  if (argc < 5) {
    cout << "Usage: ./EXEC ${DATYA_DIRECTORY} ${IMAGE_PREFIX} ${START_NUM} ${END_NUM}" << endl;
    return -1;
  }

  vector <ImageContainer*> images;
  String imageDir(argv[1]);
  String imagePrefix(argv[2]);
  int startNumber = atoi(argv[3]);
  int endNumber = atoi(argv[4]);
  
  /*
    Create response map and feature points for each images
  */
  for (int idx = startNumber; idx <= endNumber; ++idx) {  
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
  for (int idx = 0; idx < images.size(); ++idx) {
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
