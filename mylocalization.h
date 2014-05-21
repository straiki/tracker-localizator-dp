#ifndef MYLOCALIZATION_H
#define MYLOCALIZATION_H

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "myloader.h"

using namespace cv;
using namespace std;

class mylocalization
{
public:
    mylocalization(myloader *data);
    Mat calibrate(Mat);
    Mat getCameraMatrix(std::vector<Point2f>imgPoints);



    bool FindPoseEstimation(cv::Mat_<double> &rvec, cv::Mat_<double> &t, cv::Mat_<double> &R, std::vector<cv::Point3f> ppcloud, std::vector<cv::Point2f> imgPoints);
private:
    myloader *data;

    Mat _cameraMatrix;
    Mat _disortionMatrix;
};

#endif // MYLOCALIZATION_H
