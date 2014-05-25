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

//static const vector<pair<int, int> > ar_edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
//{4, 5}, {5, 6}, {6, 7}, {7, 4},
//{0, 4}, {1, 5}, {2, 6}, {3, 7},
//{1, 8}, {2, 8}, {6, 9}, {5, 9}, {8, 9}};

//static vector<Point3f> ar_verts = {
//    {0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0},
//    {0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1},
//        {0.5, 1.5, 0}, {0.5, 1.5, 1}
//};

static const vector<pair<int, int> > ar_edges = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
    {4, 8}, {7, 8}, {5, 9}, {6, 9}, {8, 9}};

static vector<Point3f> ar_verts = {
    {0.2, 1, 0}, //0
    {0, 0, 1}, //1
    {1, 0, 1}, //2
    {1.2, 1, 0}, //3
    {0.2, 2, 0}, //4
    {0, 1, 1}, //5
    {1, 1, 1}, //6
    {1.2, 2, 0}, //7
    {0.7, 2.5, 0},
    {0.5, 1.5, 1}
};


void draw_ar(Mat out, Rect tracked);
void draw_ar(Mat out, vector<Point2f> tracked_points);

vector<Point2f> tracked_points;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if ( event == EVENT_FLAG_LBUTTON ) // EVENT_FLAG_CTRLKEY
    {
        if(tracked_points.size() == 4)
            tracked_points.clear();
        cout << "Left mouse button is clicked - position (" << x << ", " << y << ")" << endl;
        Point2f newpoint = {x,y};
        tracked_points.push_back(newpoint);
    }

}

int main(int argc, char ** argv){

    VideoCapture cap;


    if(argc > 1){
        cout << argv[1] << endl;
        cap.open(argv[1]);

    }else{
        cap.open(0);
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
Mat h;
//    bool enough = false;

    int maxCorners = 300;
    int minCorners = 45;
    vector<Point2f> corners, corners_prev, corners_init;

    vector<vector<Point2f>> correspondences(2, vector<Point2f>()); // init two vector of vectors :D


    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

//    thread t1(Thread);

    namedWindow(WIN1, WINDOW_NORMAL);
    setMouseCallback(WIN1, CallBackFunc, NULL);


    Mat out;
    bool init = true, drawPoints = true;

    Size boxSize = {200,200};
    Rect box;

    box += boxSize;
    box += Point(1920/2, 1080/2);

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

        if(tracked_points.size() < 4){
            while(tracked_points.size() != 4){
                waitKey(10);

                for(Point p : tracked_points){
                    circle(out, p, 2,Scalar(255,0,0));
                }
                imshow(WIN1, out);
            }

//            for(int i = 0; i < tracked_points.size(); i ++){
//                tracked_points[i].y -= 300;
//            }
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
//            vector<Point2f> chessboardCorners;
//            findChessboardCorners(gray, Size(9,6), chessboardCorners, CV_CALIB_CB_ADAPTIVE_THRESH);

//            for(Point2f p : chessboardCorners){
//                circle(out, p, 3, Scalar(120,255,0), 2);
//            }
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(gray_prev, gray, corners_prev, corners, status, err, winSize, 3, termcrit, 0, 0.001);

            size_t i,k;

            double move_x = 0.;
            double move_y = 0.;

            int moves_counter = 0;

            correspondences[0].clear();correspondences[1].clear();
                for(i = k = 0; i < status.size(); ++i){
                    if(!status[i])
                        continue;


                    correspondences[0].push_back(corners_init[i]);
                    correspondences[1].push_back(corners[i]);
    //                corners[k++] = corners[i];
    //                corners_init[k++] = corners_init[i];

    //                circle(gray, corners[i], 3, PINK, -1, 8);
    //                circle(gray_prev, corners_prev[i], 3, GREEN, -1, 8);


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

                    if(drawPoints){
                        circle(out, corners_init[i], 3, BLUE, -1, 8);
                        circle(out, corners[i], 3, GREEN, -1, 8);
                        circle(out, corners_prev[i], 3, GREEN, -1, 8);

                        line(out, p, q, RED, 1, CV_AA, 0 );
                        p.x = (int) (q.x + 9 * cos(angle + pi / 4));
                        p.y = (int) (q.y + 9 * sin(angle + pi / 4));
                        line( out, p, q, RED, 1, CV_AA, 0 );
                        p.x = (int) (q.x + 9 * cos(angle - pi / 4));
                        p.y = (int) (q.y + 9 * sin(angle - pi / 4));
                        line( out, p, q, RED, 1, CV_AA, 0 );
                    }




                }

            if(correspondences[0].size() > 4){
                h = findHomography(correspondences[0], correspondences[1]);

                vector<Point2f> new_points(4);
                perspectiveTransform(tracked_points, new_points, h);



                line(out, new_points[0], new_points[1], Scalar(0,255,0));
                line(out, new_points[1], new_points[2], Scalar(0,255,0));
                line(out, new_points[2], new_points[3], Scalar(0,255,0));
                line(out, new_points[3], new_points[0], Scalar(0,255,0));

                draw_ar(out, new_points);
            }



//            corners_prev.resize(k);
            if(corners_prev.size() < minCorners){
                init = true;
            }


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
            case 's':
                cout << "H: " << h << endl;
                waitKey();
            break;
        }

//        t1.join();

    }

    return EXIT_SUCCESS;

}

void draw_ar(Mat out, vector<Point2f> tracked_points){

    vector<Point3f> quad_3d = {{tracked_points[0].x,tracked_points[0].y,0},{tracked_points[1].x,tracked_points[1].y,0},{tracked_points[2].x,tracked_points[2].y,0},{tracked_points[3].x,tracked_points[3].y,0}};

    double fx = 0.5 + 18 / 50;
    int w,h;
    w = 1920;h = 1080;
    Mat K = (cv::Mat_<double>(3,3) <<	fx*w ,	0	,		0.5*(w-1),
                                            0,			fx*w,	0.5*(h-1),
                                            0,			0,			1);

    Mat dist_coeff = cv::Mat_<double>::zeros(1,4);

    Vec3d tvec;
    Mat rvec;
    solvePnP(quad_3d, tracked_points, K, dist_coeff, rvec, tvec );

    vector<Point3f> in_verts;
    int counter = 0;

    for(Point3f vert : ar_verts){
        Point3f newVert = {vert.x * (tracked_points[2].x-tracked_points[0].x), vert.y *(tracked_points[2].y-tracked_points[0].y),vert.z * +(tracked_points[2].y-tracked_points[0].y)*0.5};
//        switch (counter) {
//        case 0:
//            newVert.x = vert.x * (tracked_points[0].x);
//            newVert.y = vert.y * (tracked_points[0].y);
//            newVert.z = vert.z * -(tracked_points[0].x )*0.3;
//            break;
//        case 1:
//            newVert.x = vert.x * (tracked_points[1].x);
//            newVert.y = vert.y * (tracked_points[1].y);
//            newVert.z = vert.z * -(tracked_points[0].x )*0.3;
//            break;
//        case 2:
//            newVert.x = vert.x * (tracked_points[2].x);
//            newVert.y = vert.y * (tracked_points[2].y);
//            newVert.z = vert.z * -(tracked_points[2].x )*0.3;
//            break;
//        case 3:
//            newVert.x = vert.x * (tracked_points[3].x);
//            newVert.y = vert.y * (tracked_points[3].y);
//            newVert.z = vert.z * -(tracked_points[3].x )*0.3;
//            break;
//        default:
//            counter = 0;
//            break;
//        }

        newVert += {tracked_points[0].x,tracked_points[0].y,0};
        in_verts.push_back(newVert);
        counter++;
    }

    vector<Point2f> verts;
    projectPoints(in_verts, rvec, tvec, K, dist_coeff, verts);

    for(pair<int,int> p : ar_edges){
        Point p1, p2;
        p1.x = verts[p.first].x; p1.y = verts[p.first].y;
        p2.x = verts[p.second].x; p2.y = verts[p.second].y;
        line(out, p1,p2, Scalar(255,255,0), 2);
    }

}

//    case 0:
//        newVert.x = vert.x * (tracked_points[0].x - tracked_points[1].x);
//        newVert.y = vert.y * (tracked_points[0].y - tracked_points[1].y);
//        newVert.z = vert.z * -(tracked_points[0].x - tracked_points[1].x)*0.3;
//        break;
//    case 1:
//        newVert.x = vert.x * (tracked_points[1].x - tracked_points[2].x);
//        newVert.y = vert.y * (tracked_points[1].y - tracked_points[2].y);
//        newVert.z = vert.z * -(tracked_points[0].x - tracked_points[1].x)*0.3;
//        break;
//    case 2:
//        newVert.x = vert.x * (tracked_points[2].x - tracked_points[3].x);
//        newVert.y = vert.y * (tracked_points[2].y - tracked_points[3].y);
//        newVert.z = vert.z * -(tracked_points[2].x - tracked_points[3].x)*0.3;
//        break;
//    case 3:
//        newVert.x = vert.x * (tracked_points[3].x - tracked_points[0].x);
//        newVert.y = vert.y * (tracked_points[3].y - tracked_points[0].y);
//        newVert.z = vert.z * -(tracked_points[3].x - tracked_points[0].x)*0.3;
//        break;
//    default:

//def draw_overlay(self, vis, tracked):
//       x0, y0, x1, y1 = tracked.target.rect
//       quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
//       fx = 0.5 + cv2.getTrackbarPos('focal', 'plane') / 50.0
//       h, w = vis.shape[:2]
//       K = np.float64([[fx*w, 0, 0.5*(w-1)],
//                       [0, fx*w, 0.5*(h-1)],
//                       [0.0,0.0,      1.0]])
//       dist_coef = np.zeros(4)
//       ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)
//       verts = ar_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.3] + (x0, y0, 0)
//       verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
//       for i, j in ar_edges:
//           (x0, y0), (x1, y1) = verts[i], verts[j]
//           cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)


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
