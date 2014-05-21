#include "mydetector.h"

bool sort_by_first(std::pair<int,std::pair<int,int> > a, std::pair<int,std::pair<int,int> > b) { return a.first > b.first; }

/**
 * @brief mydetector::detect Detect keypoints for both frames
 */
mydetector::mydetector(myloader *data):data(data)
{
//    _detector = FeatureDetector::create("PyramidFAST");
//    _extractor = DescriptorExtractor::create("ORB");

//    _detector = FeatureDetector::create("SURF");
//    _extractor = DescriptorExtractor::create("SURF");

}

/**
 * @brief mydetector::detect_extract detect and extract KP for both frames and keypoints
 */
void mydetector::detect_extract()
{
    initModule_nonfree();
//    std::cout << "Ima here" << std::endl;
    if(!isFrameOK()) return; // check if frame is set

    SurfFeatureDetector detector(350);
    SurfDescriptorExtractor extractor;

    detector.detect(_frame, _keypoints_1);
    extractor.compute(_frame, _keypoints_1, _descriptors_1);

//    _detector->detect(_frame, _keypoints_1);
//    _extractor->compute(_frame, _keypoints_1, _descriptors_1);

//@TODO
//    std::cout << "KP's: " << _keypoints_1.size() << std::endl;


}

/**
 * @brief mydetectro::matchFeatures only match features
 */
void mydetector::matchFeatures()
{
    unsigned int loop1_top = data->imgs.size();
    unsigned int frame_num_i = 0;

//    std::cout << "Looping for: " << loop1_top << std::endl;
//#pragma omp parallel for
    for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
//        std::cout << "Loop " << frame_num_i << std::endl;

//        std::cout << "------------ Match " << data->imgs_names[frame_num_i] << " and input.jpg \n";
        std::vector<DMatch> matches_tmp;
        match(frame_num_i,&matches_tmp);

        data->matches_matrix[std::make_pair(frame_num_i,data->imgs.size())] = matches_tmp;

        // @TODO: Dont need to do that here
        std::vector<DMatch> matches_tmp_flip = FlipMatches(matches_tmp);
        data->matches_matrix[std::make_pair(data->imgs.size(),frame_num_i)] = matches_tmp_flip;
    }

}

/**
 * @brief mydetector::match Match keypoints in last two frames using Flann based matcher
 * @return
 */
void mydetector::match(int idx, vector<DMatch> *matches)
{
    const vector<KeyPoint>& imgpts1 = data->imgpts[idx];
    const vector<KeyPoint>& imgpts2 = _keypoints_1;
    const Mat& descriptors_1 = data->descriptors[idx];
    const Mat& descriptors_2 = _descriptors_1;

    std::vector< DMatch > good_matches_,very_good_matches_;
    std::vector<KeyPoint> keypoints_1, keypoints_2;

//    stringstream ss; ss << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
//    cout << ss.str();
//    stringstream ss1; ss1 << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
//    cout << ss1.str();

    keypoints_1 = imgpts1;
    keypoints_2 = imgpts2;

    if(descriptors_1.empty()) {
        CV_Error(0,"descriptors_1 is empty");
    }
    if(descriptors_2.empty()) {
        CV_Error(0,"descriptors_2 is empty");
    }

    //matching descriptor vectors using Brute Force matcher
//    BFMatcher matcher(NORM_HAMMING,true); //allow cross-check. use Hamming distance for binary descriptor (ORB)
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches_;
    if (matches == NULL) {
        matches = &matches_;
    }
    if (matches->size() == 0) {
        matcher.match( descriptors_1, descriptors_2, *matches );
    }

    assert(matches->size() > 0);

    vector<KeyPoint> imgpts1_good,imgpts2_good;


    std::set<int> existing_trainIdx;
    for(unsigned int i = 0; i < matches->size(); i++ )
    {
        //"normalize" matching: sometimes imgIdx is the one holding the trainIdx
        if ((*matches)[i].trainIdx <= 0) {
            (*matches)[i].trainIdx = (*matches)[i].imgIdx;
        }

        if( existing_trainIdx.find((*matches)[i].trainIdx) == existing_trainIdx.end() &&
           (*matches)[i].trainIdx >= 0 && (*matches)[i].trainIdx < (int)(keypoints_2.size())
            )
        {
            good_matches_.push_back( (*matches)[i]);
            imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
            imgpts2_good.push_back(keypoints_2[(*matches)[i].trainIdx]);
            existing_trainIdx.insert((*matches)[i].trainIdx);
        }
    }

//    vector<uchar> status;
    vector<KeyPoint> imgpts2_very_good,imgpts1_very_good;

    assert(imgpts1_good.size() > 0);
    assert(imgpts2_good.size() > 0);
    assert(good_matches_.size() > 0);
    assert(imgpts1_good.size() == imgpts2_good.size() && imgpts1_good.size() == good_matches_.size());

    //Select features that make epipolar sense
    Mat F = getFundamentalMat(keypoints_1,keypoints_2,imgpts1_very_good,imgpts2_very_good,good_matches_);

#ifdef DEBUG_SHOW
    Mat img_1 = data->imgs[idx];
    Mat img_2 = _frame;
    {
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                    good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        std::stringstream ss; ss << "Feature Matches " << idx;
        imshow(ss.str() , img_matches );
        waitKey(500);
        destroyWindow(ss.str());
    }
#endif
}



///**
// * @brief mydetector::setMatchedKeyPoints sets good keypoints from matches (for single frame showing)
// */
//void mydetector::setMatchedKeyPoints()
//{
//    std::vector<KeyPoint> kp;

//    for(auto it = _good_matches.begin(); it != _good_matches.end(); it++ )
//    {
//        if(it->trainIdx >= _keypoints_2.size())
//            continue;

//        KeyPoint s = _keypoints_2.at((it->trainIdx));
//        kp.push_back(s);
//    }

//    _kp = kp;
//}

/**
 * @brief mydetector::clear
 */
void mydetector::clear()
{
    _good_matches.clear();
    _keypoints_1.clear();
}

/**
 * @brief mydetector::isFrameOK
 * @return
 */
bool mydetector::isFrameOK()
{
    return !_frame.empty();
}

/**
 * @brief mydetector::isDescriptorsOK checks if Descriptors are ok
 * @return
 */
bool mydetector::isDescriptorsOK()
{
    return !_descriptors_1.empty();
}
/**
 * @brief mydetector::isKpOk CHecks if KeyPoints Vectors are ok
 * @return
 */
bool mydetector::isKpOk()
{
    return !_keypoints_1.empty();
}

Mat mydetector::getFundamentalMat(const vector<KeyPoint>& imgpts1,
                       const vector<KeyPoint>& imgpts2,
                       vector<KeyPoint>& imgpts1_good,
                       vector<KeyPoint>& imgpts2_good,
                       vector<DMatch>& matches)
{
    //Eliminate keypoints based on the fundamental matrix
    vector<uchar> status(imgpts1.size());

    imgpts1_good.clear(); imgpts2_good.clear();

    vector<KeyPoint> imgpts1_tmp;
    vector<KeyPoint> imgpts2_tmp;
    if (matches.size() <= 0) {
        //points already aligned...
        imgpts1_tmp = imgpts1;
        imgpts2_tmp = imgpts2;
    } else {
        GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
    }

    Mat F;
    {
        vector<Point2f> pts1,pts2;
        KeyPointsToPoints(imgpts1_tmp, pts1);
        KeyPointsToPoints(imgpts2_tmp, pts2);

        double minVal,maxVal;
        minMaxIdx(pts1,&minVal,&maxVal);
        F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold [Snavely07 4.1]
    }

    vector<DMatch> new_matches;

    for (unsigned int i=0; i<status.size(); i++) {
        if (status[i])
        {
            imgpts1_good.push_back(imgpts1_tmp[i]);
            imgpts2_good.push_back(imgpts2_tmp[i]);

            if (matches.size() <= 0) { //points already aligned...
                new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
            } else {
                new_matches.push_back(matches[i]);
            }
        }
    }

//    cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
    matches = new_matches;

    return F;
}


int mydetector::findHomographyInliers2Views(int vi, int vj)
{
//    vj = 1;//i have to match only against one frame
    vector<KeyPoint> ikpts,jkpts; vector<Point2f> ipts,jpts;
    GetAlignedPointsFromMatch(data->imgpts[vi],_keypoints_1,data->matches_matrix[std::make_pair(vi,vj)],ikpts,jkpts);
    KeyPointsToPoints(ikpts,ipts); KeyPointsToPoints(jkpts,jpts);

    double minVal,maxVal; minMaxIdx(ipts,&minVal,&maxVal);

    vector<uchar> status;
    Mat H = findHomography(ipts,jpts,status,CV_RANSAC, 0.004 * maxVal);
    return countNonZero(status); //number of inliers
}

void mydetector::find2D3DCorrespondences(
    std::vector<Point3f>& ppcloud,
    std::vector<Point2f>& imgPoints)
{
    int working_view = best_id;
    ppcloud.clear(); imgPoints.clear();

    vector<int> pcloud_status(data->pcloud.size(),0);
    for (size_t i = 0; i < data->imgpts.size(); ++i )
    {
        int old_view = i;
        //check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
        std::vector<DMatch> matches_from_old_to_working = data->matches_matrix[std::make_pair(old_view,data->imgs.size())];

        for (unsigned int match_from_old_view=0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
            // the index of the matching point in <old_view>
            int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;

            //scan the existing cloud (pcloud) to see if this point from <old_view> exists
            for (unsigned int pcldp=0; pcldp<data->pcloud.size(); pcldp++) {
                // see if corresponding point was found in this point
                if (idx_in_old_view == data->pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
                {
                    //3d point in cloud
                    ppcloud.push_back(data->pcloud[pcldp].pt);
                    //2d point in image i
                    imgPoints.push_back(data->imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);

                    pcloud_status[pcldp] = 1;
                    break;
                }
            }
        }
    }
    std::cout << "found " << ppcloud.size() << " 3d-2d point correspondences"<<std::endl;
}

void mydetector::find2D3DCorrespondences2(std::vector<Point3f> &ppcloud, std::vector<Point2f> &imgPoints)
{
    // get points from matches to ppcloud
    for(unsigned int i = 0; i < _good_matches.size(); ++i)
    {
        DMatch *actual = &_good_matches[i];
        int from_picture = actual->queryIdx;
        int from_pcloud = actual->trainIdx;

        imgPoints.push_back(_keypoints_1[from_picture].pt);
        ppcloud.push_back(data->pcloud[from_pcloud].pt);

    }
    // get points from matches to imgPoints
}

void mydetector::findMatchesWithModel(){
    const Mat& descriptors_1 = _descriptors_1;
    const Mat& descriptors_2 = data->model_descriptors;

//      BFMatcher matcher(NORM_L2);
    FlannBasedMatcher matcher;
      std::vector< DMatch > matches;
      matcher.match(descriptors_1 , descriptors_2, matches );

      double max_dist = 0; double min_dist = 100;

      for( int i = 0; i < descriptors_1.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      for( int i = 0; i < descriptors_1.rows; i++ )
      { if( matches[i].distance <= max(3*min_dist, 0.02) )
        { _good_matches.push_back( matches[i]); }
      }

      printf("-- Max dist : %f \n", max_dist );
      printf("-- Min dist : %f \n", min_dist );

//      std::cout << descriptors_1.size() << std::endl;
//      std::cout << descriptors_2.size() << std::endl;
      std::cout << _good_matches.size() << std::endl;

}


void mydetector::findBestMatch(){
    //sort pairwise matches to find the best Homography inliers
    std::cout << "Find highest match...";

    unsigned int top = data->imgs.size();

    for(unsigned int num = 0; num < top; ++num) {
        std::pair<int, int> p = std::make_pair(num,data->imgs.size());
        std::vector<cv::DMatch>*i = &(data->matches_matrix[p]);
        if(i->size() < 100)
            matches_sizes.push_back(std::make_pair(100,p));
        else {
            int Hinliers = findHomographyInliers2Views(num,data->imgs.size());
            int percent = (int)(((double)Hinliers) / ((double)i->size()) * 100.0);
            std::cout << "[" << num << "," << data->imgs.size() << " = "<<percent<<"%] ";
            matches_sizes.push_back(std::make_pair((int)percent,std::make_pair(num, data->imgs.size())));
        }
    }
    std::cout << std::endl;
    matches_sizes.sort(sort_by_first);

    best_id = matches_sizes.front().second.first; // set best cam (closest)
}
