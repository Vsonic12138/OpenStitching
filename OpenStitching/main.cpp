#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>  // 使用SIFT
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

int main()
{
    std::vector<cv::Mat> images;
    fs::path img_folder = fs::current_path().parent_path() / "imgs" / "BarCode";  // 图片文件夹路径
    std::cout << "图片路径: " << img_folder << std::endl;

    if (!fs::exists(img_folder))
    {
        std::cerr << "路径不存在: " << img_folder << std::endl;
        return -1;
    }

    // 遍历文件夹中的图片
    for (const auto& entry : fs::directory_iterator(img_folder))
    {
        fs::path img_path = entry.path();
        std::cout << "检查文件: " << img_path << std::endl;

        if (img_path.extension() == ".jpg" || img_path.extension() == ".png" || img_path.extension() == ".bmp" || img_path.extension() == ".BMP")
        {
            cv::Mat img = cv::imread(img_path.string());
            if (!img.empty())
            {
                images.push_back(img);  // 加入到待拼接的图像列表中
            }
            else
            {
                std::cout << "无法读取图片: " << img_path << std::endl;
            }
        }
    }

    if (images.size() < 2)
    {
        std::cout << "需要至少两张图片进行拼接。" << std::endl;
        return -1;
    }

    // 使用SIFT特征检测
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 用于存储关键点和描述符
    std::vector<std::vector<cv::KeyPoint>> keypoints(images.size());
    std::vector<cv::Mat> descriptors(images.size());

    // 检测每张图片的关键点和描述符
    for (size_t i = 0; i < images.size(); ++i)
    {
        sift->detectAndCompute(images[i], cv::noArray(), keypoints[i], descriptors[i]);
    }

    // 使用BFMatcher进行特征点匹配
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors[0], descriptors[1], matches);

    // 对匹配结果进行排序
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
        });

    // 可视化匹配点，只显示前10个匹配点
    cv::Mat img_matches;
    cv::drawMatches(images[0], keypoints[0], images[1], keypoints[1], matches, img_matches, cv::Scalar::all(-1),
    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 显示匹配点
    //cv::imshow("Matches", img_matches);
    cv::imwrite("matches.jpg", img_matches);  // 保存匹配点结果
    cv::waitKey(0);  // 等待按键退出

    // 创建Stitcher实例,使用 SCANS 模式进行拼接、还有参数为 PANORAMA 是拼接全景图，但是速度较慢，对于平面图像的拼接效果不太好
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    // 设置特征点检测的分辨率比例（影响置信度），1表示全分辨率，值越小速度越快但可能降低精度
    stitcher->setRegistrationResol(0.8);  // 你可以调整这个值，根据需要的性能和质量来设置

    // 启用波浪校正，避免图像波浪失真问题
    stitcher->setWaveCorrection(false);

    // 拼接图片
    cv::Mat pano;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Stitcher::Status status = stitcher->stitch(images, pano);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "拼接图片耗时: " << elapsed.count() << " 秒" << std::endl;

    // 检查拼接状态
    if (status == cv::Stitcher::OK)
    {
        std::cout << "拼接成功！" << std::endl;
        fs::path output_path = img_folder / "stitched_output.bmp";
        cv::imwrite(output_path.string(), pano);  // 保存拼接结果

        // 输出拼接结果的分辨率
        int width = pano.cols;
        int height = pano.rows;
        std::cout << "拼接图像的分辨率: " << width << "x" << height << std::endl;
    }
    else
    {
        std::cout << "拼接失败，错误代码: " << status << std::endl;
    }

    // 等待用户输入
    std::cout << "按回车键继续..." << std::endl;
    std::cin.get();  // 等待用户按下回车键

    return 0;
}
