#include "k_means.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
//    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "useage: ./test_k_means input_image_path k iteration\n "
                     "example: ./test_k_means "
                     "../images/test_data/lena.png 3 10"
                  << std::endl;
        std::exit(-1);
    }

    if (img.channels() != 3) {
        std::cout << "please use a image with 3 channels";
        std::exit(-1);
    }
    cv::imshow("test",img);
    int k = strtol(argv[2], NULL, 10);
    int iteration = strtol(argv[3], NULL, 10);

    int convergence_radius = 1e-6;

    Kmeans kmeans(img, k);
    kmeans.run(iteration, convergence_radius);

    std::vector<Sample> samples = kmeans.get_result_samples();
    std::vector<Center> centers = kmeans.get_result_centers();

    cv::Mat result(img.size(), img.type());

    for (const Sample& sample : samples) {
        for (int channel = 0; channel < 3; channel++) {
            result.at<cv::Vec3b>(sample.row_, sample.col_)[channel] =
                    centers[sample.label_].feature_[channel];
        }
    }
    cv::Mat concat_img;
    cv::hconcat(img, result, concat_img);
    cv::imshow("left: original image, right: kmeans result", concat_img);
    cv::waitKey(0);
}
