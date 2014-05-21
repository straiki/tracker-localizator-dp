#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <random>

#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "Common.h"
#include "myloader.h"

using namespace std; using namespace cv;

void Thread();

#define WIN1 "Display window"

bool MainThreadDone = false;
bool DispThreadDone = false;

WriteData w;
mutex w_mutex;

static const double pi = 3.14159265358979323846;

inline static double square(int a)
{
 return a * a;
}

static const vector<pair<int, int> > ar_edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
{4, 5}, {5, 6}, {6, 7}, {7, 4},
{0, 4}, {1, 5}, {2, 6}, {3, 7},
{4, 8}, {5, 8}, {6, 9}, {7, 9}, {8, 9}};

static vector<Point3d> ar_verts = {
    {0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0},
    {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1},
        {0, 0.5, 2}, {1, 0.5, 2}
};

int main(int argc, char ** argv){

    VideoCapture cap;


    if(argc > 1){
        cout << argv[1] << endl;
        cap.open(argv[1]);
    }else{
        cap.open(1);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640.);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480.);
    }

    if(!cap.isOpened())
    {
        cerr << "Cant open video" << endl;
        return EXIT_FAILURE;
    }
    Mat frame, gray, frame_prev, gray_prev;

    cap >> frame;

//    myloader loader("input_data.yaml",frame);

//    loader.parseConfig();


    w.fps = 0;
    w.new_frame = true;

    bool enough = false;

    int maxCorners = 300;
    int minCorners = 45;
    vector<Point2f> corners, corners_prev, corners_init;

    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

//    thread t1(Thread);

    namedWindow(WIN1);

    Mat out;
    bool init = true, drawPoints = true;

    while(!DispThreadDone)
    {
        cap >> frame;

        if(frame.empty())
            break;

        cvtColor(frame, gray, CV_BGR2GRAY);

        Mat gray_blend;
        if(gray_prev.empty()){
            frame.copyTo(gray_blend);
        }
        else{
            frame.copyTo(gray_blend);
//            addWeighted(gray, 0.5, gray_prev, 0.5,0.0, gray_blend);
        }

        if (gray_blend.type()==CV_8UC1) {
                      //input image is grayscale
            cvtColor(gray_blend, out, CV_GRAY2RGB);

        } else {
            gray_blend.copyTo(out);
        }

        if(init){
            goodFeaturesToTrack(gray, corners, maxCorners, 0.01, 10.0, Mat(), 3,0, 0.04);

            if(corners.empty())
                continue;

            cornerSubPix(gray, corners, subPixWinSize, Size(-1,-1), termcrit);
            init = false;

            corners_prev.clear();
            corners_init = corners;
        }

        if(!corners_prev.empty())
        {
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(gray_prev, gray, corners_prev, corners, status, err, winSize, 3, termcrit, 0, 0.001);

            size_t i,k;
            if(drawPoints)
                for(i = k = 0; i < status.size(); ++i){
                    if(!status[i])
                        continue;
    //                corners[k++] = corners[i];
    //                corners_init[k++] = corners_init[i];

    //                circle(gray, corners[i], 3, PINK, -1, 8);
    //                circle(gray_prev, corners_prev[i], 3, GREEN, -1, 8);

                    circle(out, corners_init[i], 3, BLUE, -1, 8);
                    circle(out, corners[i], 3, GREEN, -1, 8);
                    circle(out, corners_prev[i], 3, GREEN, -1, 8);

                    Point p,q;
                    p.x = (int) corners_init[i].x;
                    p.y = (int) corners_init[i].y;
                    q.x = (int) corners_prev[i].x;
                    q.y = (int) corners_prev[i].y;

                    double angle;		angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
                    double hypotenuse;	hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );

                    /* Here we lengthen the arrow by a factor of three. */
                    q.x = (int) (p.x - hypotenuse * cos(angle));
                    q.y = (int) (p.y - hypotenuse * sin(angle));


                    line(out, p, q, RED, 1, CV_AA, 0 );

                    p.x = (int) (q.x + 9 * cos(angle + pi / 4));
                    p.y = (int) (q.y + 9 * sin(angle + pi / 4));
                    line( out, p, q, RED, 1, CV_AA, 0 );
                    p.x = (int) (q.x + 9 * cos(angle - pi / 4));
                    p.y = (int) (q.y + 9 * sin(angle - pi / 4));
                    line( out, p, q, RED, 1, CV_AA, 0 );

                }
//            corners_prev.resize(k);
            if(corners_prev.size() < minCorners)
                init = true;
        }



        imshow(WIN1,out);
//        imshow("Actual Frame", gray);
//        if(!gray_prev.empty())
//            imshow("Previous Frame", gray_prev);

        std::swap(corners_prev,corners);
        cv::swap(gray_prev, gray);



        char c = cvWaitKey(10);
        switch (c){
            case 'q':
                MainThreadDone = true; //finish him
                DispThreadDone = true;
            break;
            case 'r':
                init = true;
            break;
            case ' ':
                drawPoints = !drawPoints;
            break;
        }

//        t1.join();

    }

    return EXIT_SUCCESS;

}

//void Thread() {
//    namedWindow("matches", CV_WINDOW_AUTOSIZE);

//    std::vector<KeyPoint> kp;

//    while( !MainThreadDone )
//    {
//        double _time_in_ticks = (double)cv::getTickCount();


//        if(frame.empty() || frame_gray.empty()) continue;

//        putText(frame, "Number of corners" + to_string(w.corners.size()), cvPoint(30,90), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
//        putText(frame, "FPS "  + to_string(w.fps), cvPoint(30,110), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

//        if(!frame.empty())
//            imshow("matches", frame);
//        char c = waitKey(10);
//        switch (c){
//            case 'q':
//                MainThreadDone = true; //finish him
//            case 'n':
//                {
////                    w_mutex.lock();
//                    w.new_frame = true;
////                    w_mutex.unlock();
//                }
//            break;
//        }
//        {
////            w_mutex.lock();
//            w.fps = 1. / ((double)cv::getTickCount() - _time_in_ticks)/cv::getTickFrequency();
////            w_mutex.unlock();
//        }

//    }
//    DispThreadDone = true;

//}
////        if(!enough){
////            goodFeaturesToTrack(gray, corners, maxCorners, 0.05, 2.0);
////            cornerSubPix(gray, corners, subPixWinSize, Size(-1,-1), termcrit);
////            enough = true;

////        }

////        if(!corners_prev.empty()){

////            if(gray_prev.empty()){
////                gray.copyTo(gray_prev);
////            }

////            vector<uchar> status; vector<float> err;

////            calcOpticalFlowPyrLK(gray_prev, gray, corners_prev, corners, status, err, winSize, 3, termcrit, 0, 0.001);

////            size_t i,k;
////            for(i = k = 0; i < status.size(); ++i){
////                if(!status[i])
////                    continue;

////                corners[k++] = corners[i];

////                circle(frame, corners[i], 3, Scalar(0,255,0), -1, 8);
////            }
////            corners_prev.resize(k);

////            putText(frame,"Tracked: " + to_string(k), cvPoint(10,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

////            if(k < minCorners)
////                enough = false;
////            else
////                enough = false;
////        }

////        imshow("Video", frame);
////        char c = waitKey();
////        if(c == 'q')
////            break;

////        std::swap(corners_prev,corners);
////        cv::swap(gray_prev, gray);
