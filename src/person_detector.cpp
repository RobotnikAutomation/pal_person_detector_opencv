/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2013, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Jordi Pages <jordi.pages@pal-robotics.com> */

// PAL headers
#include <pal_detection_msgs/Detections2d.h>

// ROS headers
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>  // For pcl::fromROSMsg
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// OpenCV headers
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost headers
#include <boost/scoped_ptr.hpp>
#include <boost/foreach.hpp>

// Std C++ headers
#include <vector>

/**
 * @brief The PersonDetector class encapsulating an image subscriber and the OpenCV's CPU HOG person detector
 *
 * @example rosrun person_detector_opencv person_detector image:=/camera/image _rate:=5 _scale:=0.5
 *
 */
class PersonDetector
{
public:

  PersonDetector(ros::NodeHandle& nh,
                 ros::NodeHandle& pnh,
                 double imageScaling = 1.0, 
                 const std::string &topic = "/xtion/rgb/image_raw", 
                 const std::string &transport="raw");
  virtual ~PersonDetector();

protected:

  ros::NodeHandle _nh, _pnh;

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

  void detectPersons(const cv::Mat& img,
                     std::vector<cv::Rect>& detections);

  void scaleDetections(std::vector<cv::Rect>& detections,
                       double scaleX, double scaleY) const;

  void publishDetections(const std::vector<cv::Rect>& detections) const;

  void publishDebugImage(cv::Mat& img,
                         const std::vector<cv::Rect>& detections) const;

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

  void publishVelocity(std::vector<cv::Rect>* detections, cv_bridge::CvImageConstPtr cvImgPtr, cv::Mat* img);

  float distance_to_maintain;
  float linear_threshold;
  float angular_threshold;
  float img_width;
  float time_to_x;
  float time_to_angle;
  float gain_linear_velocity;
  float gain_angular_velocity;
  double hit_threshold;
  double final_threshold;
  double _imageScaling;
  bool use_mean_shift;
  mutable cv_bridge::CvImage _cvImgDebug;

  int y_index , z_index;
  int x_tracked , y_tracked;
  float angular_speed;
  float linear_speed;

  boost::scoped_ptr<cv::HOGDescriptor> _hogCPU;

  image_transport::ImageTransport _imageTransport, _privateImageTransport;
  image_transport::Subscriber _imageSub;
  ros::Subscriber _depthSub;
  ros::Time _imgTimeStamp;

  ros::Publisher _detectionPub;
  ros::Publisher _velocityPub;
  image_transport::Publisher _imDebugPub;

};

PersonDetector::PersonDetector(ros::NodeHandle& nh,
                               ros::NodeHandle& pnh,
                               double imageScaling, 
                               const std::string &topic, 
                               const std::string &transport):
  _nh(nh),
  _pnh(pnh),
  _imageScaling(imageScaling),
  _imageTransport(nh),
  _privateImageTransport(pnh)
{  

  _hogCPU.reset( new cv::HOGDescriptor ); 
  // _hogCPU.reset( new cv::HOGDescriptor(cv::Size(48,96), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9, 1 )  );
  _hogCPU->setSVMDetector( cv::HOGDescriptor::getDefaultPeopleDetector() );
  // _hogCPU->setSVMDetector( cv::HOGDescriptor::getDaimlerPeopleDetector() );

  image_transport::TransportHints transportHint(transport);

  _imageSub   = _imageTransport.subscribe(topic, 1, &PersonDetector::imageCallback, this, transportHint);
  _depthSub = _nh.subscribe("/robot/front_rgbd_camera/depth/color/points", 1, &PersonDetector::pointCloudCallback, this);
  // subscriber_ = nh_.subscribe(pointcloud_topic_, 1, &PointCloudProcessor::pointCloudCallback, this);
  // pnh_.subscribe<robotnik_msgs::ptz>(subscriber_rgb_command_topic_, 10, &Link750Camera::rgbCommandTopicSubCb, this);
  _imDebugPub = _privateImageTransport.advertise("debug", 10);

  _detectionPub = _pnh.advertise<pal_detection_msgs::Detections2d>("detections", 10);
  _velocityPub = _pnh.advertise<geometry_msgs::Twist>("Rcmd_vel", 10);

  _pnh.getParam("distance_to_maintain_", distance_to_maintain);
  _pnh.getParam("linear_threshold_", linear_threshold);
  _pnh.getParam("angular_threshold_", angular_threshold);
  _pnh.getParam("img_width_", img_width);
  _pnh.getParam("time_to_x_", time_to_x);
  _pnh.getParam("time_to_angle_", time_to_angle); 
  _pnh.getParam("gain_angular_velocity_", gain_angular_velocity);
  _pnh.getParam("gain_linear_velocity_", gain_linear_velocity); 
  _pnh.getParam("hit_threshold_", hit_threshold);
  _pnh.getParam("final_threshold_", final_threshold); 
  _pnh.getParam("use_mean_shift_", use_mean_shift);

  cv::namedWindow("person detections");

  y_index = 0;
  z_index = 0;

  x_tracked = 0;
  y_tracked = 0;

  angular_speed = 0.0;
  linear_speed = 0.0;
}

PersonDetector::~PersonDetector()
{
  cv::destroyWindow("person detections");
}

void PersonDetector::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  // Convert PointCloud2 to pcl::PointCloud
  // ROS_INFO_STREAM("At least entered the callback");
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);

  // ROS_INFO_STREAM("Received point with y_index: " << y_index << " and z_index: " << z_index);
  

  if (z_index*y_index >= cloud.width*cloud.height) 
  {
        // ROS_WARN("Coordinates out of bounds");
        return;
  }

    // Compute the index based on z (column) and y (row) and retrieve the x value
    int index = y_index * cloud.width + z_index;  // Adjust this based on the exact mapping in your point cloud
    float x = cloud.points[index].x;

    // ROS_INFO("Point at (y = %d, z = %d): x = %f", y_index, z_index, x);
}

void PersonDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cvImgPtr;
  cvImgPtr = cv_bridge::toCvShare(msg);

  _imgTimeStamp = msg->header.stamp;

  cv::Mat img(static_cast<int>(_imageScaling*cvImgPtr->image.rows),
              static_cast<int>(_imageScaling*cvImgPtr->image.cols),
              cvImgPtr->image.type());

  if ( _imageScaling == 1.0 )
    cvImgPtr->image.copyTo(img);
  else
  {
    cv::resize(cvImgPtr->image, img, img.size());
  }

  std::vector<cv::Rect> detections;

  detectPersons(img, detections);

  if ( _imageScaling != 1.0 )
  {
    scaleDetections(detections,
                    static_cast<double>(cvImgPtr->image.cols)/static_cast<double>(img.cols),
                    static_cast<double>(cvImgPtr->image.rows)/static_cast<double>(img.rows));
  }

  publishVelocity(&detections, cvImgPtr, &img);

  publishDetections(detections);

  cv::Mat imDebug = cvImgPtr->image.clone();
  publishDebugImage(imDebug, detections);
}

void PersonDetector::publishVelocity(std::vector<cv::Rect>* detections, cv_bridge::CvImageConstPtr cvImgPtr, cv::Mat* img)
{
  int x , y, width, height;
  float distance;
  float min_distance = 1.0;
  float angle = 0.0;

  if(detections == nullptr)
  {
    geometry_msgs::Twist move_cmd;
    move_cmd.linear.x = 0.0;
    move_cmd.angular.z = 0.0;
    _velocityPub.publish(move_cmd);
    return;
  }

  if(detections->size() == 0)
  {
    geometry_msgs::Twist move_cmd;
    move_cmd.linear.x = 0.0;
    move_cmd.angular.z = 0.0;
    _velocityPub.publish(move_cmd);
    publishDetections(*(detections));
    cv::Mat imDebug = cvImgPtr->image.clone();
    publishDebugImage(imDebug, *detections);
    return;
  }

  x_tracked = static_cast<int>(cvImgPtr->image.cols/2);
  y_tracked = static_cast<int>(cvImgPtr->image.rows/2);
  float min_pixel_distance_to_tracked = sqrt(pow(cvImgPtr->image.cols, 2) + pow(cvImgPtr->image.rows, 2));
  //estimate distance for every detection and compute the min one
  BOOST_FOREACH(const cv::Rect& roi, *detections)
  {
    x = roi.x;
    y = roi.y;
    width  = roi.width;
    height = roi.height;

    y_index = - (x + (float)cvImgPtr->image.cols/2);
    z_index = - (y + (float)cvImgPtr->image.rows/2);

    distance = 600.0 / float(height);

    float pixel_distance_to_tracked = sqrt(pow(x - x_tracked, 2) + pow(y - y_tracked, 2));
    if(pixel_distance_to_tracked < min_pixel_distance_to_tracked)
    {
      min_pixel_distance_to_tracked = pixel_distance_to_tracked;
      x_tracked = x;
      y_tracked = y;
      min_distance = distance;
      ROS_INFO_STREAM("Computed values: min_pixel_distance_to_tracked: " << min_pixel_distance_to_tracked);
    }

    angle = ((float)cvImgPtr->image.cols/2 - ((float)x_tracked + (float)width/2));
    
    cv::rectangle(*img, roi, CV_RGB(0,255,0), 2);

    ROS_INFO_STREAM("img_width: " << cvImgPtr->image.cols << "roi x: " << x << "roi width: " << width);
  }

  
  ROS_INFO_STREAM("min_distance: " << min_distance << " angle: " << angle);
  geometry_msgs::Twist move_cmd;

  // move_cmd.linear.x = 0.0;
  // move_cmd.angular.z = 0.0;

  // when you want to rotate around itself
  if (abs(angle) > angular_threshold && abs(min_distance - distance_to_maintain) <= linear_threshold)
  {
    linear_speed = 0;
    angular_speed = gain_angular_velocity*(angle / time_to_angle);
  }

  if(abs(min_distance - distance_to_maintain) <= linear_threshold)
  {
    linear_speed = 0;
  }

  if (abs(angle) < angular_threshold)
  {
    angular_speed = 0;
  }
  if (abs(min_distance - distance_to_maintain) > linear_threshold)
  {
    linear_speed = gain_linear_velocity*(min_distance - distance_to_maintain) / time_to_x;
    angular_speed = gain_angular_velocity*(angle / time_to_angle);
  }

  move_cmd.linear.x = linear_speed;
  move_cmd.angular.z = angular_speed;

  _velocityPub.publish(move_cmd);
}

void PersonDetector::scaleDetections(std::vector<cv::Rect>& detections,
                                     double scaleX, double scaleY) const
{
  BOOST_FOREACH(cv::Rect& detection, detections)
  {
    cv::Rect roi(detection);
    detection.x      = static_cast<long>(roi.x      * scaleX);
    detection.y      = static_cast<long>(roi.y      * scaleY);
    detection.width  = static_cast<long>(roi.width  * scaleX);
    detection.height = static_cast<long>(roi.height * scaleY);
  }
}


void PersonDetector::detectPersons(const cv::Mat& img,
                                   std::vector<cv::Rect>& detections)
{ 
  double start = static_cast<double>(cv::getTickCount());

  _hogCPU->detectMultiScale(img,
                            detections,
                            hit_threshold,                //hit threshold: decrease in order to increase number of detections but also false alarms
                            cv::Size(8,8),    //win stride
                            cv::Size(0,0),    //padding 24,16
                            1.02,             //scaling
                            final_threshold,                //final threshold
                            use_mean_shift);            //use mean-shift to fuse detections

  double stop = static_cast<double>(cv::getTickCount());
  ROS_DEBUG_STREAM("Elapsed time in detectMultiScale: " << 1000.0*(stop-start)/cv::getTickFrequency() << " ms");
}

void PersonDetector::publishDetections(const std::vector<cv::Rect>& detections) const
{
  pal_detection_msgs::Detections2d msg;
  pal_detection_msgs::Detection2d detection;

  msg.header.frame_id = "";
  msg.header.stamp    = _imgTimeStamp;

  BOOST_FOREACH(const cv::Rect& roi, detections)
  {
    detection.x      = roi.x;
    detection.y      = roi.y;
    detection.width  = roi.width;
    detection.height = roi.height;

    msg.detections.push_back(detection);
  }

  _detectionPub.publish(msg);
}

void PersonDetector::publishDebugImage(cv::Mat& img,
                                       const std::vector<cv::Rect>& detections) const
{
  //draw detections
  BOOST_FOREACH(const cv::Rect& roi, detections)
  {
    cv::rectangle(img, roi, CV_RGB(0,255,0), 2);
  }

  if ( img.channels() == 3 && img.depth() == CV_8U )
    _cvImgDebug.encoding = sensor_msgs::image_encodings::RGB8;

  else if ( img.channels() == 1 && img.depth() == CV_8U )
    _cvImgDebug.encoding = sensor_msgs::image_encodings::MONO8;
  else
    throw std::runtime_error("Error in Detector2dNode::publishDebug: only 24-bit BGR or 8-bit MONO images are currently supported");

  _cvImgDebug.image = img;
  sensor_msgs::Image imgMsg;
  imgMsg.header.stamp = _imgTimeStamp;
  _cvImgDebug.toImageMsg(imgMsg); //copy image data to ROS message

  _imDebugPub.publish(imgMsg);
}

int main(int argc, char **argv)
{
  ros::init(argc,argv,"pal_person_detector_opencv"); // Create and name the Node
  ros::NodeHandle nh, pnh("~");

  ros::CallbackQueue cbQueue;
  nh.setCallbackQueue(&cbQueue);

  double scale = 1.0;
  pnh.param<double>("scale",   scale,    scale);

  double freq = 10;
  pnh.param<double>("rate",   freq,    freq);

  std::string imTransport = "raw";
  pnh.param<std::string>("transport",   imTransport,    imTransport);

  std::string topic = "/xtion/rgb/image_raw";
  pnh.param<std::string>("image", topic, topic);

  ROS_INFO_STREAM("Setting image scale factor to: " << scale);
  ROS_INFO_STREAM("Setting detector max rate to:  " << freq);
  ROS_INFO_STREAM("Image type:  " << imTransport);
  ROS_INFO(" ");

  ROS_INFO_STREAM("Creating person detector ...");

  PersonDetector detector(nh, pnh, scale, topic, imTransport);

  ROS_INFO_STREAM("Spinning to serve callbacks ...");

  ros::Rate rate(freq);
  while ( ros::ok() )
  {
    cbQueue.callAvailable();
    rate.sleep();
  }

  return 0;
}
