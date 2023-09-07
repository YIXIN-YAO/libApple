#include "libApple.h"
#include "AppleDetect.h"

static AppleDetect *detect = nullptr;

bool runtime_start(const char *model, const char *device) {
    if (detect == nullptr)
    {
        detect = new AppleDetect();
    }

    if (model != nullptr)
    {
        if (detect->register_model(model) < 0)
        {
            return false;
        }
    }

    if (device != nullptr)
    {
        if (detect->register_cam(device) < 0)
        {
            return false;
        }
    }
    return true;
}


int detect_single_frame(const char *device, std::vector<WORLD_COORDINATES> &data_vec, Image &image) {
    if (detect == nullptr)
    {
        return -1;
    }
    return detect->get_apple(device, data_vec, image);

}

float detect_tree_distance(const char *device) {
    if (detect == nullptr)
    {
        return -1;
    }
    return detect->get_distance(device);
}

int stop_cam(const char *device) {
    if (detect == nullptr)
    {
        return -1;
    }
    return detect->unregister_cam(device);
}

int runtime_stop() {
    if (detect == nullptr)
    {
        return -1;
    }
    delete detect;
    detect = nullptr;
    return 0;

}
