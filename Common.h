/**************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#pragma once

//#pragma warning(disable: 4244 18 4996 4800)

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <list>
#include <set>
#include <stdlib.h>     /* strtoull */

#define USE_PROFILING 1


#define RED Scalar(0,0,255)
#define BLUE Scalar(255,0,0)
#define GREEN Scalar(0,255,0)
#define BLACK Scalar(255,255,255)
#define WHITE Scalar(0,0,0)
#define GRAY Scalar(127,127,127)
#define PINK Scalar(127,0,255)
#define YELLOW Scalar(0,246,255)
#define ORANGE Scalar(0,56,255)

struct WriteData {
    int fps;
    bool new_frame;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> corners_prev;
    cv::Mat frame;
    cv::Mat frame_gray;
};


struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};

void MyFilledCircle(cv::Mat, cv::Point);
std::vector<cv::DMatch> FlipMatches(const std::vector<cv::DMatch>& matches);
void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2);

void drawArrows(cv::Mat& frame, const std::vector<cv::Point2f>& prevPts, const std::vector<cv::Point2f>& nextPts, const std::vector<uchar>& status, const std::vector<float>& verror, const cv::Scalar& line_color = cv::Scalar(0, 0, 255));


//    std::cout << msg << ">>BEGIN" << std::endl;

#ifdef USE_PROFILING
#define CV_PROFILE(msg,code)	{\
	double __time_in_ticks = (double)cv::getTickCount();\
	{ code }\
    std::cout << "\t" << std::setw(30) << msg  << "\t" << ((double)cv::getTickCount() - __time_in_ticks)/cv::getTickFrequency() << "s" << std::endl;\
}
#else
#define CV_PROFILE(msg,code) code
#endif

void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names, double downscale_factor);
void imshow_250x250(const std::string& name_, const cv::Mat& patch);
//
void readVectorOfVector(cv::FileStorage &fns, std::string name, std::vector<std::vector<cv::KeyPoint>> &vov);
void writeVectorOfVector(cv::FileStorage &fs, std::string name, std::vector<std::vector<cv::KeyPoint>> &vov);
