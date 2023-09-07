#ifndef YOLOX_REALSENSE_H
#define YOLOX_REALSENSE_H

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
#include <string>
#include "common.h"
#include <exception>

class Realsense {
    enum {
        SUCCESS = 0,
        OPEN_ERROR = -1,
        READ_ERROR = -2,
    };
    const std::string m_device;
    const int m_color_width = 1280;
    const int m_color_height = 800;
    const int m_depth_width = 1280;
    const int m_depth_height = 720;
    const int m_fps = 30;
    float m_depth_scale = 1.0;
    rs2::pipeline *mp_pipe = nullptr;
    rs2_intrinsics *mp_intrinsic_depth;
    rs2_intrinsics *mp_intrinsic_color;

public:
    explicit Realsense(const char *device, int color_width = 1280, int color_height = 800, int depth_width = 1280,
                       int depth_height = 720,
                       int fps = 30) :
            m_device(device),
            m_color_width(color_width), m_color_height(color_height),
            m_depth_width(depth_width), m_depth_height(depth_height),
            m_fps(fps), m_depth_scale(1.0), mp_pipe(nullptr),
            mp_intrinsic_depth(new rs2_intrinsics),
            mp_intrinsic_color(new rs2_intrinsics) {}

    int open() {
        rs2::config cfg;
        cfg.enable_device(m_device);
        cfg.enable_stream(RS2_STREAM_COLOR, m_color_width, m_color_height, RS2_FORMAT_BGR8, m_fps);
        cfg.enable_stream(RS2_STREAM_DEPTH, m_depth_width, m_depth_height, RS2_FORMAT_Z16, m_fps);
        try
        {
            mp_pipe = new rs2::pipeline;
            rs2::pipeline_profile profile = mp_pipe->start(cfg);
            auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            *mp_intrinsic_depth = depth_stream.get_intrinsics();
            *mp_intrinsic_color = color_stream.get_intrinsics();
            m_depth_scale = get_depth_scale(profile.get_device());
        }
        catch (std::exception &e)
        {
            if (mp_pipe != nullptr)
            {
                delete mp_pipe;
                mp_pipe = nullptr;

            }
            return OPEN_ERROR;
        }

        return SUCCESS;

    }

    int read(cv::Mat &image_color,cv::Mat &image_depth,int& width,int& height) {
        if (nullptr == mp_pipe)
        {
            return READ_ERROR;
        }
        rs2::frameset data = mp_pipe->wait_for_frames(); // Wait for next set of frames from the camera
        //Align to depth
        rs2::align align_to_depth(RS2_STREAM_COLOR);
        data = align_to_depth.process(data);
        rs2::frame color_frame = data.get_color_frame();
        rs2::frame depth_frame = data.get_depth_frame();
        const int depth_w = depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h = depth_frame.as<rs2::video_frame>().get_height();
        width = depth_frame.as<rs2::video_frame>().get_width();
        height = depth_frame.as<rs2::video_frame>().get_height();

        image_depth = cv::Mat(cv::Size(depth_w, depth_h), CV_16U, (void *) depth_frame.get_data(), cv::Mat::AUTO_STEP);
        image_color = cv::Mat(cv::Size(width, height), CV_8UC3, (void *) color_frame.get_data(), cv::Mat::AUTO_STEP);
        return SUCCESS;
    }

    const rs2_intrinsics *get_depth_intrinsic() const { return mp_intrinsic_depth; }

    const rs2_intrinsics *get_colir_intrinsic() const { return mp_intrinsic_color; }

    float get_scale() const { return m_depth_scale; }


};


#endif //YOLOX_REALSENSE_H
