#include "myloader.h"

using namespace cv;
using namespace std;

myloader::myloader(string filename, Mat *in_img) : in_img(in_img)
{
    fs.open(filename, FileStorage::READ);

    if(!fs.isOpened())
        throw new Exception();

}

bool myloader::parseConfig()
{
//    cout << "=====================================" << endl;
//    cout << "Parsing...." << endl;


    ostringstream oss;
    FileNode fn;

    std::vector<std::string> imgs_names;
    {
        fs["directory"] >> directory_;
        fs["files"] >> imgs_names;
    }

    {
        readVectorOfVector(fs,"imgpts",imgpts);
        readVectorOfVector(fs, "imgpts_good", imgpts_good);


        fn = fs["descriptors"];
        read(fn, descriptors);

//        for(size_t i = 0; i < imgpts.size(); ++i){
//            cout << "Imgpts " << imgs_names[i] << ":" << imgpts[i].size() << endl;
//        }
//        for(size_t i = 0; i < imgpts_good.size(); ++i){
//            cout << "Imgpts_good " << imgs_names[i] << ":" << imgpts_good[i].size() << endl;
//        }
    }

    size_t counter = 0;

    fn = fs["cameras"];
    if (fn.empty()){
        cerr << "Error parsing! No cameras loaded" << endl;
        return false;
    }
    {//////////////////////// cameras //////////////////////////
        vector<Mat> tmp;
        FileNodeIterator current = fn.begin(), it_end = fn.end(); // Go through the node
        for (; current != it_end; ++current)
        {
            Mat tMat;
            FileNode item = *current;
            read(item, tMat);
            tmp.push_back(tMat);
        }

        for(unsigned int i = 0; i < tmp.size(); ++i){
            Matx34d tmp_matx;
            for(int row = 0; row < tmp[i].rows; ++row){
                for(int col = 0; col < tmp[i].cols; ++ col){
                    tmp_matx(row,col) = tmp[i].at<double>(row, col);
                }
            }
            cameras.push_back(tmp_matx);
        }

        if(cameras.size() <= 0)
        {
            cerr << "Error parsing! No cameras loaded" << endl;
            return false;
        }
//        cout << "cameras: " << cameras.size() << endl;
    }
    {/// cloudpoint
        string tStr;
        read(fs["pcloud"], tStr,"0");

        counter = strtoull(tStr.c_str(), NULL, 0);

        Point3d tP;
        vector<int> tVec;
        double tErr;

        double maxX =0, maxY=0, maxZ = 0;

        for(size_t i = 0; i < counter; ++i){
            oss << "pc_pt_" << i;
            fs[oss.str()] >> tP;
            oss.str("");

            oss << "pc_img_pt_" << i;
            fs[oss.str()] >> tVec;
            oss.str("");

            oss << "pc_reproj_err_" << i;
            fs[oss.str()] >> tErr;
            oss.str("");
            i++;

            CloudPoint tCp = {tP,tVec,tErr};
            int tCounter=0;
            for(unsigned int i = 0; i < tVec.size(); ++i)
            if(tVec[i] != -1) ++tCounter;

            if(tCounter >= 2)
                pcloud.push_back(tCp);

            maxX =  fmax((double)maxX,abs(tP.x));
            maxY =  fmax((double)maxY,abs(tP.y));
            maxZ =  fmax((double)maxZ,abs(tP.z));

        }

        cout << "Cloudpoint: " << pcloud.size() << endl;
//        cout << "Max: " << maxX << " " << maxY << " " << maxZ << endl;
    }
    //load calibration matrix
    cv::FileStorage fs;
    if(fs.open(directory_+ "\\out_camera_data.yaml",cv::FileStorage::READ)) {
        fs["camera_matrix"]>>cam_matrix;
        fs["distortion_coefficients"]>>distortion_coeff;
    } else {
        //no calibration matrix file - mockup calibration
        cv::Size imgs_size = in_img->size();
        double max_w_h = MAX(imgs_size.height,imgs_size.width);
        cam_matrix = (cv::Mat_<double>(3,3) <<	max_w_h ,	0	,		imgs_size.width/2.0,
                                                0,			max_w_h,	imgs_size.height/2.0,
                                                0,			0,			1);
        distortion_coeff = cv::Mat_<double>::zeros(1,4);
    }

    K = cam_matrix;
    invert(K, Kinv); //get inverse of camera matrix

    distortion_coeff.convertTo(distcoeff_32f,CV_32FC1);
    K.convertTo(K_32f,CV_32FC1);

    return true;
}

double downscale_factor = 1.0;

void myloader::loadFiles()
{
//    cout << "Loading images " << endl;
    // open images from directory
    std::vector<cv::Mat> imgs_;

    char *cstr = new char[directory_.length() + 1];
    strcpy(cstr, directory_.c_str());

    open_imgs_dir(cstr, imgs_, imgs_names, downscale_factor);
    delete [] cstr;

    for (unsigned int i=0; i<imgs_.size(); i++) {
        imgs_orig.push_back(cv::Mat_<cv::Vec3b>());
        if (!imgs_[i].empty()) {
            if (imgs_[i].type() == CV_8UC1) {
                cvtColor(imgs_[i], imgs_orig[i], CV_GRAY2BGR);
            } else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
                imgs_[i].convertTo(imgs_orig[i],CV_8UC3,255.0);
            } else {
                imgs_[i].copyTo(imgs_orig[i]);
            }
        }

        imgs.push_back(cv::Mat());
        cvtColor(imgs_orig[i],imgs[i], CV_BGR2GRAY);

        // gray colour only

        imgpts.push_back(std::vector<cv::KeyPoint>());
        imgpts_good.push_back(std::vector<cv::KeyPoint>());
//        std::cout << ".";
    }
//    std::cout << std::endl;



//    for(std::vector<std::string>::iterator i = imgs_names.begin(); i != imgs_names.end(); ++i)
//    {
//        std::cout << *i << std::endl;
//    }


}

void myloader::createModelDescriptors()
{
    model_descriptors.create((int)pcloud.size(), 64, CV_32F);

    for(unsigned int row = 0; row < pcloud.size(); ++row){
        CloudPoint *tmp = &pcloud[row];
        for(unsigned int img = 0; img < imgs.size(); ++img){
            int idx = -1;
            if((idx = tmp->imgpt_for_img[img]) != -1){
                descriptors[img].row(idx).copyTo(model_descriptors.row(row));
                break;
            }
            continue;
        }
    }
}


