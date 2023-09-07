#ifndef YOLOX_YOLOX_H
#define YOLOX_YOLOX_H

#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class YoloXDetect {
    struct GridAndStride {
        int grid0 = 0;
        int grid1 = 0;
        int stride = 0;

        GridAndStride(int grid0, int grid1, int stride) : grid0(grid0), grid1(grid1), stride(stride) {}
    };

    const float NMS_THRESH = 0.45;
    const float BBOX_CONF_THRESH = 0.3;
    const int NUM_CLASSES = 1;
    const char *INPUT_BLOB_NAME = "images";
    const char *OUTPUT_BLOB_NAME = "output";
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    int out_size = 1;
    Logger *ptr_Logger = nullptr;
    nvinfer1::IRuntime *ptr_runtime = nullptr;
    nvinfer1::ICudaEngine *ptr_engine = nullptr;
    nvinfer1::IExecutionContext *ptr_context = nullptr;
public:
    YoloXDetect() : ptr_Logger(new Logger()) {}

    bool init(const char *trt_engine) {
        try{
            size_t size = 0;
            std::ifstream file(trt_engine, std::ios::binary);
            char *trtModelStream = nullptr;
            if (file.good())
            {
                file.seekg(0, std::ifstream::end);
                size = file.tellg();
                file.seekg(0, std::ifstream::beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, (int) size);
                file.close();
            }
            cudaSetDevice(0);
            ptr_runtime = nvinfer1::createInferRuntime(*ptr_Logger);
            if(nullptr == ptr_runtime){
                return false;
            }
            ptr_engine = ptr_runtime->deserializeCudaEngine(trtModelStream, size);
            if(nullptr == ptr_engine){
                return false;
            }
            ptr_context = ptr_engine->createExecutionContext();
            if(nullptr == ptr_context){
                return false;
            }
            delete[] trtModelStream;
            auto out_dims = ptr_engine->getBindingDimensions(1);
            for (int j = 0; j < out_dims.nbDims; j++)
            {
                out_size *= out_dims.d[j];
            }
            return true;
        }
        catch (std::exception& e){
            return false;
        }
    }

    void doInference(nvinfer1::IExecutionContext &context, float *input,
                     float *output, int output_size, const cv::Size &input_shape);

    static void fillData(const cv::Mat &img, float *blob, int channels = 3) {
        int img_h = img.rows;
        int img_w = img.cols;
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < img_h; h++)
            {
                for (int w = 0; w < img_w; w++)
                {
                    blob[c * img_w * img_h + h * img_w + w] =
                            (float) img.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }

    cv::Mat resize_image(cv::Mat &img, int cols, int rows) const {
        if (cols == INPUT_W && rows == INPUT_H)
        {
            return img;
        }
        float r = std::min((float) INPUT_W / cols, (float) INPUT_H / rows);
        int pad_w = (int) (r * (float) cols);
        int pad_h = (int) (r * (float) rows);
        cv::Mat re(pad_h, pad_w, CV_8UC3);
        cv::resize(img, re, re.size());
        cv::Mat pad(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(pad(cv::Rect(0, 0, re.cols, re.rows)));
        return pad;
    }


    static void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                                          std::vector<GridAndStride> &grid_strides) {
        for (auto stride : strides)
        {
            int num_grid = target_size / stride;
            for (int g1 = 0; g1 < num_grid; g1++)
            {
                for (int g0 = 0; g0 < num_grid; g0++)
                {
                    grid_strides.emplace_back(g0, g1, stride);
                }
            }
        }
    }

    void generate_yolox_proposals(std::vector<GridAndStride> grid_strides,
                                  float *feat_blob, float prob_threshold,
                                  std::vector<Object> &objects) const;

    static inline float intersection_area(const Object &a, const Object &b) {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    static void nms_sorted_bboxes(const std::vector<Object> &faceobjects,
                                  std::vector<int> &picked,
                                  float nms_threshold);

    static void qsort_descent_inplace(std::vector<Object> &objects);

    static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);

    void decode_outputs(float *prob, std::vector<Object> &objects,
                        float scale, int img_w, int img_h) const;

    void forward(cv::Mat image, std::vector<Object> &objects) {

        auto *prob = new float[out_size];
        int cols = image.cols;
        int rows = image.rows;
        cv::Mat re_image = resize_image(image, cols, rows);
        auto *blob = new float[re_image.total() * 3];
        fillData(re_image, blob);
        float scale = std::min((float) INPUT_W / (float) cols, (float) INPUT_H / (float) rows);
        auto start = std::chrono::system_clock::now();
        doInference(*ptr_context, blob, prob, out_size, re_image.size());
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        decode_outputs(prob, objects, scale, cols, rows);
        delete[] blob;
        delete[] prob;
    }

    virtual ~YoloXDetect() {
        if(ptr_context) ptr_context->destroy();
        if(ptr_engine) ptr_engine->destroy();
        if(ptr_runtime) ptr_runtime->destroy();
        delete ptr_Logger;
    }


};


#endif //YOLOX_YOLOX_H
