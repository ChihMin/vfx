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
#include <deque>
#include <algorithm>
#include <omp.h>
#include <sys/sysinfo.h>

using namespace cv;
using namespace std;

#define PI std::acos(-1) 

#define FOCAL_SCALE 1
#define SOBEL_KERNEL_SIZE 3
#define LENGTH 27
#define BLOCK_NUM 27
#define KNN_LEVEL 1
#define ERROR_RATE 40
#define THRESHOLD_SCALE 1.1

#define ANGLE_THRESHOLD cos(5*PI/180)
#define LENGTH_THRESHOLD 0.1

typedef vector<pair<Point, Point>> EdgeType;

class ImageContainer {
public:
  ImageContainer() { }
  ImageContainer(const Mat& _image, const Mat& _radMap, const vector<Point>& _pointVec) {
    _image.copyTo(this->image);
    _radMap.copyTo(this->radMap);
    this->pointVec.assign(_pointVec.begin(), _pointVec.end());
  }
  
  ImageContainer(const Mat& _image, const Mat& _radMap, 
                 const Mat& _Ix, const Mat& _Iy, 
                 const vector<Point>& _pointVec, string _itemName) {
    _image.copyTo(this->image);
    _radMap.copyTo(this->radMap);
    _Ix.copyTo(this->Ix);
    _Iy.copyTo(this->Iy);
    this->pointVec.assign(_pointVec.begin(), _pointVec.end());
    this->patches = this->getWindowsOfFeatures();
    this->itemName = _itemName;
  }
  
  Mat getImage() { return this->image; }
  Mat getRadMap() { return this->radMap; }
  Mat getIx() { return this->Ix; }
  Mat getIy() { return this->Iy; }
  vector<Point> getPoints() { return this->pointVec;}
  vector<Mat> getPatches() { return this->patches; } 
  string getItemName() { return this->itemName; }

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
    // cout << min << " " << max << endl;
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
          if(origX >= 0 && origX < image.rows)
            if (origY >= 0 && origY < image.cols)
              window.at<Vec3b>(matX, matY) = this->image.at<Vec3b>(origX, origY);
        }
      
      Mat patch = Mat::zeros(blockNum, blockNum, CV_8UC3);
      int blockLength = length / blockNum;
      for (int u = 0; u < blockNum; ++u)
        for (int v = 0; v < blockNum; ++v) {
          int xx = u * blockLength;
          int yy = v * blockLength;

          // Maybe here shold be modified for capture the features
          Mat block = window(Rect(xx, yy, blockLength, blockLength));
          int numOfElemts = blockLength * blockLength;
          uchar B = sum(block)[0] / numOfElemts;
          uchar G = sum(block)[1] / numOfElemts;
          uchar R = sum(block)[2] / numOfElemts;
          patch.at<Vec3b>(u, v) = Vec3b(B, G, R);
        }
      // GaussianBlur(patch, patch, Size(3,3), 0, 0, BORDER_DEFAULT);
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

  EdgeType getMatchEdges(ImageContainer *otherImage) {
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
    
    // A lambda function for patches convertion
    // Normalize 
    auto trainDataCvt = [](const vector<Mat>& patchList, 
                           int totalBlock) -> pair<Mat, Mat> {
      Mat Data(patchList.size(), totalBlock, CV_32FC1);
        Mat Labels(patchList.size(), 1, CV_32SC1);
        for (int idx = 0; idx < patchList.size(); ++idx) {
        Mat patch;
        cvtColor(patchList[idx], patch, CV_BGR2GRAY);
        // normalize(patch, patch, 255, 0, NORM_MINMAX);
        patch.convertTo(patch, CV_32FC1, 1.0, 0);
        Scalar mean, stddev;
        meanStdDev(patch, mean, stddev);
        subtract(patch, mean, patch);
        divide(patch, stddev, patch);
        // normalize(patch, patch, 255, 0, NORM_MINMAX);
        
        Labels.at<int>(idx, 0) = idx;
        // Copy all features to train data matrix
        for (int i = 0; i < patch.rows; ++i)
          for (int j = 0; j < patch.cols; ++j) {
            int curPos = i * patch.rows + j;
            Data.at<float>(idx, curPos) = patch.at<float>(i, j);
          }
      }
      pair<Mat, Mat> dataSet(Data, Labels);
      return dataSet;
    };
    
    auto matchFeatures = [](const pair<Mat, Mat>& A, 
                            const pair<Mat, Mat>& B) -> map<int, int> {
      Mat matchB2A, matchA2B;
      map <int, int> matchTable, B2ATable, A2BTable;
      int knnLevel = KNN_LEVEL;
      
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
    
    // table format = [Label B] --> Label A
    map <int, int> matchTable = matchFeatures(ASet, BSet); 
    map <int, int>::iterator it;
    
    Mat imageA = this->getFeatureImage();
    Mat imageB = otherImage->getFeatureImage();
    Mat matArray[] = { imageA, imageB };
    Mat outputMat;
    hconcat(matArray, 2, outputMat);
    vector<pair<Point, Point>> lines;
     
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
      // cout << "error rate = " << error << endl; 
      
      if (error < ERROR_RATE) {
        // line(outputMat, Point(Ay, Ax), Point(By, Bx), Scalar(0, 255, 0), 2); 
        lines.push_back(pair<Point, Point>(Point(Ax, Ay), Point(Bx, pointB[labelB].y)));
      }
    }
    
    map<int, vector<pair<Point, Point>>> vectorTable;
    int maxMatchIdx = 0;
    int maxMatchPoints = 0;
    for (int i = 0; i < lines.size(); ++i) {
      vectorTable[i] = vector<pair<Point, Point>>();
      for (int j = 0; j < lines.size(); ++j) {
        // vector A - B
        float ux = lines[i].first.x - lines[i].second.x;
        float uy = lines[i].first.y - lines[i].second.y;
        float vx = lines[j].first.x - lines[j].second.x;
        float vy = lines[j].first.y - lines[j].second.y;
        float uu = sqrt(ux * ux + uy * uy);
        float vv = sqrt(vx * vx + vy * vy);
        float dot = ux * vx + uy * vy;
        float cosTheta = dot / (uu * vv);
        float diff = sqrt((uu - vv) * (uu - vv)) / uu;
        
        if (cosTheta >= ANGLE_THRESHOLD && diff <= LENGTH_THRESHOLD)
          vectorTable[i].push_back(lines[j]);
      }
      
      int numOfPoints = vectorTable[i].size();
      if (numOfPoints > maxMatchPoints) {
        maxMatchIdx = i;
        maxMatchPoints = numOfPoints;
      }
    }
    
    vector<pair<Point, Point>> maxMatch = vectorTable[maxMatchIdx];
    vector<pair<Point, Point>>::iterator matchIt;
    for (matchIt = maxMatch.begin(); matchIt != maxMatch.end(); ++matchIt) {
      Point A = matchIt->first;
      Point B = matchIt->second;
      line(outputMat, Point(A.y, A.x), Point(B.y + imageA.cols, B.x), Scalar(0, 255, 0)); 
    }
     
    // imshow("Window", outputMat);
    // waitKey(0);
    return maxMatch;
  }
  
private:
  Mat image;
  Mat radMap;
  Mat Ix, Iy;
  vector<Point> pointVec;
  vector<Mat> patches;
  string itemName;
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
  Sobel(grayImage, DX,  -1, 1, 0, SOBEL_KERNEL_SIZE, scale, delta, BORDER_DEFAULT);
  Sobel(grayImage, DY,  -1, 0, 1, SOBEL_KERNEL_SIZE, scale, delta, BORDER_DEFAULT);
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
  
  int i, j;
  int numOfCores = get_nprocs_conf();
  #pragma omp parallel num_threads(16) private(i)
  { 
    #pragma omp for schedule(dynamic, 1)
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
  // cout << min << " " << max << endl; 
  float sumOfRadmap = sum(radMap)[0];
  float threshold = THRESHOLD_SCALE * sumOfRadmap / radMap.rows / radMap.cols;

  for (int i = 1; i < radMap.rows - 1; ++i)
    for (int j = 1; j < radMap.cols - 1; ++j)
      if (radMap.at<float>(i, j) > threshold) {
        // Find local maximum in the middle of window
        float wMin, wMax;
        int top = i - 1;
        int left = j - 1;
        int length = 3;
        bool isLarger = false;
        bool isLess = false;
        Mat window = radMap(Rect(left, top, length, length));
        for (int u = 0; u < length; ++u) 
          for (int v = 0; v < length; ++v) {
            if (u == 1 && v == 1)
              continue;
            if (window.at<float>(u, v) >  window.at<float>(1, 1)) 
              isLess = true;
          }
        
        if (!isLess)
          KillMap.at<uchar>(i, j) = 1;
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

typedef pair<vector<ImageContainer*>, map<ImageContainer*, Point>> StitchingType;
StitchingType featureMatching(const vector<ImageContainer*>& images) {
  map<ImageContainer*, map<ImageContainer*, EdgeType>> edgeTable; 
  int numOfImages = images.size();
  
  // Create feature table bewteen each images
  for (auto imageA: images) 
    for (auto imageB: images) {
        if (imageA == imageB) 
          continue; 
      
      if (edgeTable.find(imageA) == edgeTable.end())
        edgeTable[imageA] = map<ImageContainer*, EdgeType>();
      edgeTable[imageA][imageB] = imageA->getMatchEdges(imageB); 
    }

  map<ImageContainer*, bool> inList;
  deque<ImageContainer*> imageList;
  map<ImageContainer*, Point> imageShiftTable;
  ImageContainer *startImage = images[0];
  imageList.push_front(startImage);
  inList[startImage] = true;
  
  cout << "Start find neighborhood ... " << endl;
  for (int STAT = 0; STAT <= 1; ++STAT) { 
    ImageContainer* imageA = NULL;
    ImageContainer *imageType = startImage;
    int uSum = 0;
    int vSum = 0;
    while (imageA != imageType) {
      imageA = imageType;
      for (auto imageB: images) {
        if (inList.find(imageB) != inList.end())
          continue;

        EdgeType& edges = edgeTable[imageA][imageB];
        if (edges.size() <= 10)
          continue;
        
        int u = 0, v = 0;
        for (auto edge: edges) {
          Point pointA = edge.first;
          Point pointB = edge.second;
          int xx = pointA.x - pointB.x;
          int yy = pointA.y - pointB.y;
          u += xx;
          v += yy;
        }
        u = (int)((float)u / (float)edges.size() + 0.5);
        v = (int)((float)v / (float)edges.size() + 0.5);
        uSum += u;
        vSum += v; 
        
        // For LHS image searching
        // For RHS image searching
        // There is End of Image in current seatching
        bool hasNextImage = true;
        if (!STAT && v < 0)
          imageList.push_front(imageB);
        else if (STAT && v > 0)
          imageList.push_back(imageB);
        else
          continue;
        
        // accumulate current vector from first image 
        imageShiftTable[imageB] = Point(uSum, vSum); 
        inList[imageB] = true;
        imageType = imageB;
        break;
      }
    }
  }
  cout << "End of finding neighborhood .. " << endl;
  vector<Mat> stitchingImages;
  vector<ImageContainer*> outputImageList;
  deque<ImageContainer*>::iterator it;
  for (it = imageList.begin(); it != imageList.end(); ++it) {
    ImageContainer *newImage = *it;
    stitchingImages.push_back(newImage->getFeatureImage());
    outputImageList.push_back(newImage);
  }
/* 
  Mat outputMat;
  Mat* imageArray = stitchingImages.data();
  hconcat(imageArray, stitchingImages.size(), outputMat);
  imwrite("output.jpg", outputMat);
  imshow("outimage", outputMat);
  waitKey(0); 
  for (auto imageA: images)
    for (auto imageB: images)
      cout  << imageA->getItemName() << " " 
            << imageB->getItemName() << " " 
            << edgeTable[imageA][imageB].size() << endl;
*/
  return StitchingType(outputImageList, imageShiftTable);
}

void imageBlending(StitchingType& bundle) {
  vector<ImageContainer*>& imageList =  bundle.first;
  map<ImageContainer*, Point>& imageShiftTable = bundle.second;
  
  // Find offset for stitching image
  int offsetX = 0, offsetY = 0;
  int marginX = 0, marginY = 0;
  for (auto& elemt: imageShiftTable) {
    ImageContainer* image = elemt.first;
    Point shift = elemt.second;
    offsetX = min(offsetX, shift.x);
    offsetY = min(offsetY, shift.y);
  }
  
  // Find new margin for stitching images 
  Mat tempImage = imageList[0]->getFeatureImage();
  int cols = tempImage.cols;
  int rows = tempImage.rows; 
  for (auto image: imageList) {
    Point shift = imageShiftTable[image];
    int newX = shift.x - offsetX;
    int newY = shift.y - offsetY;
    marginX = max(marginX, newX + rows);
    marginY = max(marginY, newY + cols); 
  }

  Mat output = Mat::zeros(marginX, marginY, CV_8UC3);
  ImageContainer *lastImage = NULL;
  for (auto image: imageList) {
    Mat imageMat = image->getImage();
    Point shift = imageShiftTable[image];
    int newX = shift.x - offsetX;
    int newY = shift.y - offsetY;
    
    for (int i = newX; i < newX + rows; ++i)
      for (int j = newY; j < newY + cols; ++j) {
        // Do linear interpolation
        if (!lastImage) {
          output.at<Vec3b>(i, j) = imageMat.at<Vec3b>(i-newX, j-newY);
          continue;
        }
        
        int leftBound =  newY;
        int rightBound = imageShiftTable[lastImage].y - offsetY + cols;
        if (j < leftBound || j >= rightBound) {
          output.at<Vec3b>(i, j) = imageMat.at<Vec3b>(i-newX, j-newY);
        } else {
          float beta = (float)(j - leftBound) / (rightBound - leftBound);
          float alpha = 1 - beta;
          for (int rgb = 0; rgb < 3; ++rgb) {
            uchar summary = (uchar)(alpha * output.at<Vec3b>(i, j)[rgb]) + 
              (uchar)(beta * imageMat.at<Vec3b>(i-newX, j-newY)[rgb]);
            output.at<Vec3b>(i, j)[rgb] = summary;
          }
        }
      }
    
    lastImage = image;
  }
  imwrite("output.jpg", output);
  imshow("output", output);
  waitKey(0);
}


int main( int argc, char** argv )
{

  if (argc < 2) {
    cout << "Usage: ./EXEC ${DATYA_DIRECTORY} " << endl;
    return -1;
  }
  
  vector <ImageContainer*> images;
  map<string, float> focalTable;

  String imageDir(argv[1]);
  String imagePrefix(argv[2]);
  String pano = imageDir + "/" + "pano.txt";
  ifstream fin;
  fin.open(pano.c_str());
  
  auto getStringTokens = [](const string& s, char delim) -> vector<string> {
    stringstream ss(s);
    string item;
    vector<string> elems;
    while (getline(ss, item, delim)) {
      elems.push_back(item);
    }
    return elems;
  };

  vector<string> imageNameList;
  string filePath;
  float focal;
  
  fin >> filePath >> focal;
  while (!fin.eof()) {
    vector<string> tokens = getStringTokens(filePath, '\\');
    string itemName = tokens[tokens.size() - 1];
    focalTable[itemName] = focal;
    imageNameList.push_back(itemName);
    fin >> filePath >> focal;
  }
  
  int idx = 0;
  // Create response map and feature points for each images
  for (auto itemName: imageNameList) {  
    // if (idx++ == 5) break; 
    String imagePath = imageDir + "/" + itemName;
    cout << imagePath << endl;
    cout << itemName << " " << focalTable[itemName] << endl;
    Mat inImage, image;
    inImage = imread(imagePath, IMREAD_COLOR); // Read the file
    if(inImage.empty()) {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
    
    Mat projImage = Mat::zeros(inImage.rows, inImage.cols, CV_8UC3);
    int offsetXLBound = (inImage.cols - 1) / 2;
    int offsetXRBound = inImage.cols - 1 - offsetXLBound;
    int offsetYLBound = (inImage.rows - 1) / 2;
    int offsetYRBound = inImage.rows - offsetYLBound - 1;
    
    float f = focalTable[itemName];
    float S = FOCAL_SCALE * f;
    for (int y = -offsetYLBound; y <= offsetYRBound; ++y)
      for (int x = -offsetXLBound; x <= offsetXRBound; ++x) {
        float theta = fastAtan2(x, f);
        int xx = S * tan(theta * PI / 180) + offsetXLBound + 0.5;
        int yy = S * (y / sqrt((x*x + f*f))) + offsetYLBound + 0.5;
        int origY = y + offsetYLBound;
        int origX = x + offsetXLBound; 
        projImage.at<Vec3b>(yy, xx) = inImage.at<Vec3b>(origY, origX);
      }
    // imshow("proj", projImage);
    // waitKey(0);
    inImage = projImage; 
     
    Mat radMap, Ix, Iy;
    vector<Point> pointVec;
    inImage.convertTo(image, CV_32F, 1.0/255, 0) ; 
    radMap = getRadianceMap(image, Ix, Iy, 2, 0.04);
    pointVec = findFeaturePoints(radMap);
    ImageContainer *newImage = new ImageContainer(
      inImage, radMap, Ix, Iy, pointVec, itemName
    ); 
    images.push_back(newImage);
  }
  
  StitchingType bundle = featureMatching(images);
  imageBlending(bundle);
   
  return 0;
}
