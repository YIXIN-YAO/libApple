#include "AppleDetect.h"

int AppleDetect::get_apple(const char *device, std::vector<WORLD_COORDINATES> &data_vec, Image &image,
                           int width, int height, double max_radius) {
    try
    {
        int data_size = 0;
        Realsense *cam = cam_map[device];
        if (nullptr == cam || nullptr == mp_yolox)
        {
            printf("please do runtime_start before do detect\n");
            return ERROR;
        }


        cv::Mat image_color, image_depth;
        int frame_width, frame_height;
        cam->read(image_color, image_depth, frame_width, frame_height);


        cv::Mat roi = image_color(cv::Rect2i(frame_width / 2 - width / 2,
                                             frame_height / 2 - height / 2, width, width));
        std::vector<Object> objects;
        mp_yolox->forward(roi, objects);

        //处理数据
        if (!objects.empty())
        {
            float pixel[2] = {0.0, 0.0};
            float world[3] = {0, 0, 0};
            for (auto &obj : objects)
            {
                obj.rect.x += (float) frame_width / 2 - (float) width / 2;
                obj.rect.y += (float) frame_height / 2 - (float) height / 2;
                pixel[0] = (obj.rect.x + obj.rect.width / 2);
                pixel[1] = (obj.rect.y + obj.rect.height / 2);
                int x = (int) pixel[0];
                int y = (int) pixel[1];
                cv::Mat hsv, mask, element;
                cv::cvtColor(image_color, hsv, CV_BGR2HSV);
                cv::inRange(hsv, cv::Vec3d(30, 0, 0), cv::Vec3d(125, 255, 255), mask);
                element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                cv::dilate(mask, mask, element);
                image_color.setTo(0, mask);
                float depth = depth_filter(image_depth, mask, x, y, (int) obj.rect.width / 4,
                                           (int) obj.rect.height / 4, cam->get_scale());
                rs2_deproject_pixel_to_point(world, cam->get_depth_intrinsic(), pixel, depth);


                float fxy = (cam->get_colir_intrinsic()->fx + cam->get_colir_intrinsic()->fy) / 2;
                float apple_width = depth * obj.rect.width / fxy;
                float apple_height = depth * obj.rect.height / fxy;
                //过滤不满足尺寸的结果
                if (depth < 0.1 || depth > 2.5 ||
                    apple_width > max_radius * 2 / 1000 ||
                    apple_height > max_radius * 2 / 1000)
                {
                    obj.prob = 0.0;
                    continue;
                }

                float offset = (apple_width + apple_height) /
                               (4 * sqrt(world[0] * world[0] + world[1] * world[1] + world[2] * world[2]));
                world[0] = world[0] * (1 + offset);
                world[1] = world[1] * (1 + offset);
                world[2] = world[2] * (1 + offset);
                data_vec.emplace_back(world[0], world[1], world[2], obj.prob);
                obj.world = cv::Point3d(world[0], world[1], world[2]);
                data_size++;
            }
            draw_objects(image_color, objects);
        }
        cv::Mat save_image = roi.clone();
        cv::resize(save_image, save_image, cv::Size(360, 360));
        image.col = save_image.cols;
        image.row = save_image.rows;
        cv::cvtColor(save_image, save_image, CV_BGR2RGB);
        image.data = new unsigned char[image.col * image.row * 3];
        memcpy(image.data, save_image.data, image.col * image.row * 3);
        return data_size;
    }
    catch (std::exception &e)
    {
        return ERROR;
    }

}

float AppleDetect::get_distance(const char *device, int block_size, int max_h) {
    Realsense *cam = cam_map[device];
    if (nullptr == cam)
    {
        printf("cam %s not start\n", device);
        return ERROR;
    }
    cv::Mat image_depth, image_color;
    int width, height;
    if (cam->read(image_color, image_depth, width, height) < 0)
    {
        printf("read cam: %s error\n", device);
        return ERROR;
    }
    //对深度图像进行滤波处理获取有效的点，将图像下半部分去除只保留上半部分
    /*
     * 60 X 60 的区域转化为一个点
     */
    std::vector<float> block_depths;
    for (int w = 0; w + block_size < width; w += block_size)
    {
        for (int h = 0; h + block_size < max_h; h += block_size)
        {
            cv::Rect roi(w, h, block_size, block_size);
            cv::Mat image = image_depth(roi);
            //计算有效点个数,深度在0.5 - 2之间
            int valid_num = 0;
            float sum = 0;
            for (int x = 0; x < block_size; ++x)
            {
                for (int y = 0; y < block_size; ++y)
                {
                    float depth = (float) image_depth.at<uint16_t>(h + y,
                                                                   w + x) * cam->get_scale();

                    if (depth > 0.5 && depth < 2.0)
                    {
                        valid_num++;
                        sum += depth;
                    }
                }
            }
            //如果小于一半有效点，则滤除该区域
            if (2 * valid_num < block_size * block_size)
            {
                continue;
            }
            block_depths.push_back(sum / (float) valid_num);

            //计算区域的平均深度
        }
    }
    //如果有效点个数小于10
    if (block_depths.size() < 10)
    {
        return 0;
    }
    std::sort(block_depths.begin(),block_depths.end());
    float sum = 0;
    for (float d : block_depths)
    {
        sum += d;
    }
    return sum / (float) block_depths.size();
}