#ifndef MYTRACKER_H
#define MYTRACKER_H

#include <ios>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class mytracker
{
private:

public:
    mytracker();

    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> corners_prev;
};

#endif // MYTRACKER_H
