#include <ros/ros.h>
#include <ros/package.h>
#include "mtf_bridge/SharedImageReader.h"
#include "mtf_bridge/PatchTrackers.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// C++
#include <vector>
#include <math.h>
#include <assert.h>
#include <Eigen/Core>

// Tracking Framework
#include "MTF.h"

int const rate = 30;

cv::Mat display_frame;
cv::Mat gray_frame;
cv::Mat rgb_frame;
std::vector<cv::Point> new_tracker_points;
std::vector<mtf::TrackerBase*> trackers;
ros::Publisher tracker_pub;
SharedImageReader *image_reader;

using namespace mtf::params;

typedef struct Patch {
    Patch(std::vector<cv::Point> points) {
        assert(points.size() == 4);
        for(unsigned long i = 0; i < points.size(); ++i) {
            corners[i] = points[i];
        }
    }
    cv::Point2d operator[](int i) { return corners[i];}
    cv::Point2d corners[4];
} Patch;

cv::Point get_patch_center(mtf::TrackerBase &tracker) {
    Eigen::Vector3d tl(tracker.cv_corners[0].x, tracker.cv_corners[0].y, 1);
    Eigen::Vector3d tr(tracker.cv_corners[1].x, tracker.cv_corners[1].y, 1);
    Eigen::Vector3d br(tracker.cv_corners[2].x, tracker.cv_corners[2].y, 1);
    Eigen::Vector3d bl(tracker.cv_corners[3].x, tracker.cv_corners[3].y, 1);

    Eigen::Vector3d center_vec = center_vec = tl.cross(br).cross(tr.cross(bl));

    cv::Point center;
    center.x = center_vec(0) / center_vec(2);
    center.y = center_vec(1) / center_vec(2);
    return center;
}

// TODO: Should we remove the gaussian blur?
void format_frame() {
    image_reader->get_next_frame()->convertTo(rgb_frame, rgb_frame.type());
    cv::cvtColor(rgb_frame, gray_frame, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray_frame, gray_frame, cv::Size(5, 5), 3);
}

mtf_bridge::Patch get_tracker_patch(mtf::TrackerBase &tracker) {
    mtf_bridge::Point top_left;
    top_left.x = tracker.cv_corners[0].x;
    top_left.y = tracker.cv_corners[0].y;

    mtf_bridge::Point top_right;
    top_right.x = tracker.cv_corners[1].x;
    top_right.y = tracker.cv_corners[1].y;

    mtf_bridge::Point bot_right;
    bot_right.x = tracker.cv_corners[2].x;
    bot_right.y = tracker.cv_corners[2].y;

    mtf_bridge::Point bot_left;
    bot_left.x = tracker.cv_corners[3].x;
    bot_left.y = tracker.cv_corners[3].y;

    mtf_bridge::Patch patch;
    patch.corners[0] = top_left;
    patch.corners[1] = top_right;
    patch.corners[2] = bot_right;
    patch.corners[3] = bot_left;

    cv::Point center_point = get_patch_center(tracker);

    mtf_bridge::Point center;
    center.x = center_point.x;
    center.y = center_point.y;
    patch.center = center;

    return patch;
}

void update_trackers() {
    if (trackers.empty()) {
        return;
    }

    format_frame();

    mtf_bridge::PatchTrackers tracker_msg;
    for(std::vector<mtf::TrackerBase*>::const_iterator tracker = trackers.begin(); tracker != trackers.end(); ++tracker) {
        (*tracker)->update();
        mtf_bridge::Patch tracker_patch = get_tracker_patch(**tracker);
        tracker_msg.trackers.push_back(tracker_patch);
    }
    tracker_pub.publish(tracker_msg);
}

void draw_patch(const cv::Point2d* corners, cv::Scalar color) {
    line(display_frame, corners[0], corners[1], color);
    line(display_frame, corners[1], corners[2], color);
    line(display_frame, corners[2], corners[3], color);
    line(display_frame, corners[3], corners[0], color);
}

void draw_frame(std::string cv_window_title) {
    cv::cvtColor(*(image_reader->get_next_frame()), display_frame, cv::COLOR_RGB2BGR);

    // Draw trackers
    if (!trackers.empty()) {
        for(std::vector<mtf::TrackerBase*>::const_iterator tracker = trackers.begin(); tracker != trackers.end(); ++tracker) {
            draw_patch((*tracker)->cv_corners, cv::Scalar(0, 0, 255));
            cv::Point center = get_patch_center(**tracker);
            cv::circle(display_frame, center, 5, cv::Scalar(0, 0, 255), -1);
            // Black outline
            cv::circle(display_frame, center, 5, cv::Scalar(0, 0, 0), 2);
        }
    }

    // Show construction of tracker
    for (std::vector<cv::Point>::const_iterator point = new_tracker_points.begin(); point != new_tracker_points.end(); ++point) {
        // White filled circle
        cv::circle(display_frame, *point, 5, cv::Scalar(255, 255, 255), -1);
        // Black outline
        cv::circle(display_frame, *point, 5, cv::Scalar(0, 0, 0), 2);
    }
    imshow(cv_window_title, display_frame);
    char key = cv::waitKey(10);

    if (key == 'd') {
        if (trackers.size() > 0) {
            delete(*trackers.begin());
            trackers.erase(trackers.begin());
        }
    }
}

// Assume points are ordered: 1 ---- 2
//                            |      |
//                            4 ---- 3
void initialize_tracker() {
    cv::Mat new_tracker_corners(2, 4, CV_64FC1);
    new_tracker_corners.at<double>(0, 0) = new_tracker_points[0].x;
    new_tracker_corners.at<double>(1, 0) = new_tracker_points[0].y;
    new_tracker_corners.at<double>(0, 1) = new_tracker_points[1].x;
    new_tracker_corners.at<double>(1, 1) = new_tracker_points[1].y;
    new_tracker_corners.at<double>(0, 2) = new_tracker_points[2].x;
    new_tracker_corners.at<double>(1, 2) = new_tracker_points[2].y;
    new_tracker_corners.at<double>(0, 3) = new_tracker_points[3].x;
    new_tracker_corners.at<double>(1, 3) = new_tracker_points[3].y;

    mtf::ImgParams *init_params = new mtf::ImgParams(50, 50, gray_frame);
    mtf::TrackerBase *new_tracker = getTrackerObj(mtf_sm, mtf_am, mtf_ssm, init_params);

    format_frame();
    new_tracker->initialize(new_tracker_corners);

    trackers.push_back(new_tracker);
    ROS_INFO_STREAM("Tracker initialized");
    draw_frame("TrackingNode");
}

void mouse_cb(int mouse_event, int x, int y, int flags, void* param) {
    // Right mouse click restarts setting of tracker points
    if (mouse_event == CV_EVENT_RBUTTONUP) {
        new_tracker_points.clear();
        return;
    }

    if (mouse_event == CV_EVENT_LBUTTONUP) {
        ROS_DEBUG_STREAM("Click at x: " << x << ", " << y);
        new_tracker_points.push_back(cv::Point(x,y));
        if (new_tracker_points.size() == 4) {
            initialize_tracker();
            new_tracker_points.clear();
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "trackingNode");
    ros::NodeHandle nh_("~");

    // Initialize input_obj
    std::string tracker_params;
    nh_.param<std::string>("tracker_params", tracker_params, "/cfg/sampel_tracker_params.cfg");
    std::string params_path = ros::package::getPath("mtf_bridge") + tracker_params;
    int fargc = readParams((char *)params_path.c_str());
    parseArgumentPairs(fargv, fargc);

    image_reader = new SharedImageReader();

    // Initialize OpenCV window and mouse callback
    std::string cv_window_title = "TrackingNode";
	cv::namedWindow(cv_window_title, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(cv_window_title, mouse_cb);

    tracker_pub = nh_.advertise<mtf_bridge::PatchTrackers>("patch_tracker", 1);

	ros::Rate loop_rate(rate);

    while(!image_reader->is_initialized()) {
        ROS_INFO_STREAM("Waiting while system initializes");
        ros::spinOnce();
        ros::Duration(0.7).sleep();
    }

    ROS_INFO_STREAM("Left click to select tracker corners. Right click to reset corners, press 'd' to delet tracker");

    // Initialize frame
    rgb_frame.create(image_reader->get_height(), image_reader->get_width(), CV_32FC3);
    gray_frame.create(image_reader->get_height(), image_reader->get_width(), CV_32FC1);

	while(ros::ok()){
		ros::spinOnce();
        update_trackers();
        draw_frame(cv_window_title);
		loop_rate.sleep();
    }
    return 0;
}
