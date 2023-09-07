#ifndef YOLOX_APPLEDETECT_H
#define YOLOX_APPLEDETECT_H

#include <unordered_map>
#include <string>
#include "Realsense.h"
#include "yolox.h"
#include "libApple.h"


class AppleDetect {
    enum {
        ERROR = -1,
        SUCCESS = 0
    };
    std::unordered_map<std::string, Realsense *> cam_map;
    YoloXDetect *mp_yolox = nullptr;

public:
    AppleDetect() : mp_yolox(new YoloXDetect())  {}

    //模型注册
    int register_model(const char *model) {
        if (!mp_yolox->init(model))
        {
            return ERROR;
        }
        return SUCCESS;
    }

    //摄像头注册
    int register_cam(const char *device) {
        auto* cam = new Realsense(device);
        if(cam->open() < 0){
            delete cam;
            return ERROR;
        }
        cam_map.insert({device,cam });
        return SUCCESS;
    }

    int unregister_cam(const char *device) {
        auto iter = cam_map.find(device);
        if(iter == cam_map.end()){
            return ERROR;
        }
        delete iter->second;
        cam_map.erase(iter);
        return SUCCESS;
    }
    //摄像头根据ID获取
    Realsense *get_cam(const char *device) {
        if (cam_map.find(device) == cam_map.end())
        {
            return nullptr;
        }
        return cam_map[device];
    }

    int get_apple(const char *device, std::vector<WORLD_COORDINATES> &data_vec, Image &image,
               int width = 640, int height = 640, double max_radius = 75.0);

    float get_distance(const char* device, int block_size=60 ,int max_h=600);


    ~AppleDetect(){
        if(mp_yolox){
            delete mp_yolox;
            mp_yolox = nullptr;
        }
        auto iter = cam_map.begin();
        for(;iter != cam_map.end();++iter){
            delete iter->second;
            iter->second = nullptr;
        }
    }
};


#endif //YOLOX_APPLEDETECT_H
