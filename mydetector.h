#ifndef MYDETECTOR_H
#define MYDETECTOR_H

#include <thread>
#include <mutex>
#include <utility>      // std::pair

#include <ios>

//#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "myloader.h"

using namespace cv;

class mydetector
{

public:
    mydetector(myloader *data);

    void setFrame(Mat &frame){ this->_frame = frame; }

    void match(int idx, vector<DMatch> *matches); // count of matched features
    inline std::vector<DMatch> getMatches(){ return _good_matches;}
    void switchFrames();

    /**
     * @brief mydetector::getKeypoints
     * @return
     */
    std::vector<KeyPoint> getKeypoints() { return _keypoints_1; }
    Mat getDescriptors() { return _descriptors_1; }

    void clear();
    void setMatchedKeyPoints();

    void detect_extract();
    void matchFeatures();
    Mat getFundamentalMat(const vector<KeyPoint> &imgpts1, const vector<KeyPoint> &imgpts2, vector<KeyPoint> &imgpts1_good, vector<KeyPoint> &imgpts2_good, vector<DMatch> &matches);
    int findHomographyInliers2Views(int vi, int vj);
    void find2D3DCorrespondences(std::vector<Point3f> &ppcloud, std::vector<Point2f> &imgPoints);
    void find2D3DCorrespondences2(std::vector<Point3f> &ppcloud, std::vector<Point2f> &imgPoints);
    void findBestMatch();

    int best_id;

    void findMatchesWithModel();
private:
    bool isFrameOK();
    bool isDescriptorsOK();
    bool isKpOk();

    myloader *data;

    Ptr<FeatureDetector> _detector;
    Ptr<DescriptorExtractor> _extractor;

    Mat _frame, _prevFrame;
    std::vector<DMatch> _matches;
    std::vector<KeyPoint> _keypoints_1;
    Mat _descriptors_1;
    std::vector<DMatch> _good_matches;

    std::list<std::pair<int,std::pair<int,int> > > matches_sizes;

    double _min_dist = 200;



};

#endif // MYDETECTOR_H
