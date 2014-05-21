#include "mylocalization.h"
#include "mydetector.h"
#include "Common.h"


mylocalization::mylocalization(myloader *data) : data(data)
{ // init matrices
    cv::FileStorage fs;
    string filename = "./camera_calib_lumia.yaml";
    fs.open(filename,cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open " << filename << endl;
        throw new Exception();
    }

//    std::cout << "Reading from file..." << std::endl;
//    fs["Camera_Matrix"]>>_cameraMatrix;
//    fs["Distortion_Coefficients"]>>_disortionMatrix;

        _cameraMatrix = (Mat_<double>(3,3) << 9.3917748355209255e+02, 0., 2.8581296802660052e+02, 0., 9.3917748355209255e+02, 3.4586465094599612e+02, 0., 0., 1.);
        _disortionMatrix = (Mat_<double>(5,1) << -5.6697603665763464e-02, 1.4614933207282410e-01, -2.0794949372678544e-02, -2.9613956143842737e-02, 1.1742032167200889e-01);
}

Mat mylocalization::calibrate(Mat input)
{
    Mat tmp = input.clone();
//    undistort(input, tmp, _cameraMatrix, _disortionMatrix);
    return tmp;
}

Mat mylocalization::getCameraMatrix(std::vector<Point2f>imgPoints)
{
    std::vector<Point3f> objPoints;
    for(auto &point : imgPoints){
        Point3f tmp = Point3f(point.x, point.y, 0);
        objPoints.push_back(tmp);
    }
    Mat rvec, tvec;
    solvePnP(objPoints, imgPoints, _cameraMatrix, _disortionMatrix, rvec, tvec);

    Mat out;
    Rodrigues(rvec, out);
    out = out.t(); //rotace inverze
    tvec = -out * tvec;

    Mat T(4,4, out.type());
    T(Range(0,3), Range(0,3)) = out * 1;
    T(Range(0,3), Range(3,4)) = tvec * 1;
    double *p = T.ptr<double>(3);
    p[0] = p[1] = p[2] = 0; p[3] = 1;

    return T;
}

bool mylocalization::FindPoseEstimation(
    cv::Mat_<double>& rvec,
    cv::Mat_<double>& t,
    cv::Mat_<double>& R,
    std::vector<cv::Point3f> ppcloud,
    std::vector<cv::Point2f> imgPoints
    )
{
    if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "Not enough cloud points... (only " << ppcloud.size() << ")" <<endl;
        return false;
    }


    vector<int> inliers;

    double minVal,maxVal; cv::minMaxIdx(imgPoints,&minVal,&maxVal);


    CV_PROFILE("Finding pose: solvePnPRansac",cv::solvePnPRansac(ppcloud, imgPoints, data->K, data->distortion_coeff, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);)
   //CV_PROFILE("solvePnP",cv::solvePnP(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, CV_EPNP);)


    vector<cv::Point2f> projected3D;
    cv::projectPoints(ppcloud, rvec, t, data->K, data->distortion_coeff, projected3D);

    if(inliers.size()==0) { //get inliers
        for(unsigned int i=0;i<projected3D.size();i++) {
            if(norm(projected3D[i]-imgPoints[i]) < 10.0)
                inliers.push_back(i);
        }

    }

    double sum_error = 0.;
    for(unsigned int j=0;j<inliers.size();j++) {
        int i = inliers[j];
        if(norm(projected3D[i]-imgPoints[i]) < 10.0){
            sum_error += pow(projected3D[i].x - imgPoints[i].x,2)+pow(projected3D[i].y - imgPoints[i].y,2);
        }
    }
    cout << "ReprojectionError >> " /*<< "[" << sum_error << "/" << inliers.size() << "] "*/ << sqrt(sum_error/inliers.size()) << endl;

    if(inliers.size() < (double)(imgPoints.size())/5.0) {
//        cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
        return false;
    }

//    if(cv::norm(t) > 200.0) {
//        // wont work goodly
//        cerr << "Estimated camera movement is too big, skip this camera\r\n";
//        return false;
//    }

    cv::Rodrigues(rvec, R);
//    if(!CheckCoherentRotation(R)) {
//        cerr << "Rotation is incoherent. Should try a different base view..." << endl;
//        return false;
//    }

//    std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
    return true;
}

