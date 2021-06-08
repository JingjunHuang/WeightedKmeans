#include "k_means.h"
#include <algorithm>
#include <vector>


static std::random_device rd;
static std::mt19937 rng(rd());

/**
 * @brief get_random_index, check_convergence, calc_square_distance are helper
 * functions, you can use it to finish your homework:)
 *
 */

std::set<int> get_random_index(int max_idx, int n);

float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers);

inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2);

/**
 * @brief Construct a new Kmeans object
 *
 * @param img : image with 3 channels
 * @param k : wanted number of cluster
 */
Kmeans::Kmeans(cv::Mat img, const int k) {
    centers_.resize(k);
    last_centers_.resize(k);
    samples_.reserve(img.rows * img.cols);
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            std::array<float, 3> tmp_feature;
            for (int channel = 0; channel < 3; channel++) {
                tmp_feature[channel] =
                        static_cast<float>(img.at<cv::Vec3b>(r, c)[channel]);
            }
            samples_.emplace_back(tmp_feature, r, c, -1);
        }
    }
}

/**
 * @brief initialize k centers randomly, using set to ensure there are no
 * repeated elements
 *
 */
void Kmeans::initialize_centers() {
    std::set<int> random_idx =
            get_random_index(samples_.size() - 1, centers_.size());
    int i_center = 0;

    for (auto index : random_idx) {
        centers_[i_center].feature_ = samples_[index].feature_;
        i_center++;
    }


    auto step = 10;
    std::vector<std::vector<std::vector<int>>> rangeCount;
    int a[26][26][26];

    for(int i = 0;i<26;i++){
        for(int j = 0;j<26;j++){
            for(int k = 0;k<26;k++){
                a[i][j][k] = 0;
            }
        }
    }

    for(Sample &sample:samples_){
        auto i = (int)sample.feature_[0]/step;
        auto j = (int)sample.feature_[1]/step;
        auto k = (int)sample.feature_[2]/step;
        a[i][j][k]++;
    }



//e.g. step == 10 ,  rangeCount[3][4][5] == 50
//    rangeCount[10][10][10] = 10;
//  概率密度法：
//    auto radius = 255 /  centers_.size();
//    std::cout<<"1"<<std::endl;
//    for(int k = 0;k<centers_.size();k++){
//        std::vector<int> countList;
//        std::vector<std::vector<int>> map;
//        std::cout<<"2"<<std::endl;
//        for(auto i = samples_.begin(); i != samples_.end(); i++) {
//            auto count = 0;
//            std::vector<int> sampleInRadius;
//            for(auto j = samples_.begin(); j != samples_.end(); j++) {
//                std::cout<<i - samples_.begin()<<":"<<j - samples_.begin()<<std::endl;
//                if(calc_square_distance(i->feature_,j->feature_) <= radius && j->label_ != -1){
//                    count+=1;
//                    sampleInRadius.push_back(j - samples_.begin());
//                }
//            }
//            countList.push_back(count);
//            map.push_back(sampleInRadius);
//        }
//        std::vector<int>::iterator maxDenSample = std::max_element(std::begin(countList), std::end(countList));
//        auto maxDenSampleIndex = std::distance(std::begin(countList), maxDenSample);
//        for(int filteredSampleIndex:map[maxDenSampleIndex]){
//            samples_[filteredSampleIndex].label_ = k;
//        }
//    }
//    update_centers();

}

/**
 * @brief change the label of each sample to the nearst center
 *
 */
void Kmeans::update_labels() {
    for (Sample& sample : samples_){
        // TODO update labels of each feature
        float minDistance = FLT_MAX;
        for(int i = 0;i<centers_.size();i++){
            if (calc_square_distance(centers_[i].feature_, sample.feature_) < minDistance) {
                minDistance = calc_square_distance(centers_[i].feature_, sample.feature_);
                sample.label_ = i;
            }
        }
    }
}

/**
 * @brief move the centers according to new lables
 *
 */
void Kmeans::update_centers() {
    last_centers_ = centers_;
    for(int k = 0;k<centers_.size();k++){

        std::vector<Sample> Filtered;
        for(Sample &sample:samples_){
            if(sample.label_ == k){
                Filtered.push_back(sample);
            }
        }


        std::array<float,3> total{};
        for(Sample& filtered:Filtered){
            for(int i = 0;i<3;i++)
                total[i] += filtered.feature_[i];
        }
        std::array<float,3> mean{};
        for(int i = 0;i<3;i++){
            mean[i] = total[i] / Filtered.size();
        }
        centers_[k].feature_ = mean;
    }
}

/**
 * @brief check terminate conditions, namely maximal iteration is reached or it
 * convergents
 *
 * @param current_iter
 * @param max_iteration
 * @param smallest_convergence_radius
 * @return true
 * @return false
 */
bool Kmeans::is_terminate(int current_iter, int max_iteration,
                          float smallest_convergence_radius) const {
    if (current_iter >= max_iteration || check_convergence(centers_,last_centers_)<= smallest_convergence_radius) {
        return true;
    }else{
        return false;
    }
}



std::vector<Sample> Kmeans::get_result_samples() const {
    return samples_;
}
std::vector<Center> Kmeans::get_result_centers() const {
    return centers_;
}
/**
 * @brief Execute k means algorithm
 *                1. initialize k centers randomly
 *                2. assign each feature to the corresponding centers
 *                3. calculate new centers
 *                4. check terminate condition, if it is not fulfilled, return
 *                   to step 2
 * @param max_iteration
 * @param smallest_convergence_radius
 */
void Kmeans::run(int max_iteration, float smallest_convergence_radius) {
    initialize_centers();
    int current_iter = 0;
    while (!is_terminate(current_iter, max_iteration,
                         smallest_convergence_radius)) {
        current_iter++;
        update_labels();
        update_centers();
    }
}

/**
 * @brief Get n random numbers from 1 to parameter max_idx
 *
 * @param max_idx
 * @param n
 * @return std::set<int> A set of random numbers, which has n elements
 */
std::set<int> get_random_index(int max_idx, int n) {
    std::uniform_int_distribution<int> dist(1, max_idx + 1);

    std::set<int> random_idx;
    while (random_idx.size() < n) {
        random_idx.insert(dist(rng) - 1);
    }
    return random_idx;
}
/**
 * @brief Calculate the L2 norm of current centers and last centers
 *
 * @param current_centers current assigned centers with 3 channels
 * @param last_centers  last assigned centers with 3 channels
 * @return float
 *
 */
float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers) {
    float convergence_radius = 0;
    for (int i_center = 0; i_center < current_centers.size(); i_center++) {
        convergence_radius +=
                calc_square_distance(current_centers[i_center].feature_,
                                     last_centers[i_center].feature_);
    }
    return convergence_radius;
}

/**
 * @brief calculate L2 norm of two arrays
 *
 * @param arr1
 * @param arr2
 * @return float
 */
inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2) {
    return std::pow((arr1[0] - arr2[0]), 2) + std::pow((arr1[1] - arr2[1]), 2) +
           std::pow((arr1[2] - arr2[2]), 2);
}



