#ifndef MYLOADER_H
#define MYLOADER_H

#include "opencv2/opencv.hpp"
#include "Common.h"

class myloader
{
private:
    cv::FileStorage fs;

    cv::Mat* in_img;

public:
    myloader(std::string filename, cv::Mat *in_img);
    bool parseConfig();
    void loadFiles();

    ////////////////////////////////////////
    std::vector<std::vector<cv::KeyPoint> > imgpts;
    std::vector<std::vector<cv::KeyPoint> > imgpts_good;

    /*std::vector<*/std::vector<cv::Mat> /*>*/ descriptors;

    std::vector<cv::Mat_<cv::Vec3b> > imgs_orig;
    std::vector<cv::Mat> imgs;
    std::vector<std::string> imgs_names;
    std::string directory_;

    std::vector<cv::Matx34d> cameras;

    std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;

    cv::Mat K;
    cv::Mat_<double> Kinv;

    cv::Mat cam_matrix,distortion_coeff;
    cv::Mat distcoeff_32f;
    cv::Mat K_32f;

    std::map<int,cv::Matx34d> Pmats;

    std::vector<CloudPoint> pcloud;

    std::vector<int> indexes_desc_points;
    cv::Mat model_descriptors;

    cv::Matx34d getCam(int i)
    {
        return cameras[i];
    }
    void createModelDescriptors();
};
#endif // MYLOADER_H
