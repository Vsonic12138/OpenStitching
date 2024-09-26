#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>  // ʹ��SIFT
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
    fs::path img_folder = fs::current_path().parent_path() / "imgs" / "BarCode";  // ͼƬ�ļ���·��
    std::cout << "ͼƬ·��: " << img_folder << std::endl;

    if (!fs::exists(img_folder))
    {
        std::cerr << "·��������: " << img_folder << std::endl;
        return -1;
    }

    // �����ļ����е�ͼƬ
    for (const auto& entry : fs::directory_iterator(img_folder))
    {
        fs::path img_path = entry.path();
        std::cout << "����ļ�: " << img_path << std::endl;

        if (img_path.extension() == ".jpg" || img_path.extension() == ".png" || img_path.extension() == ".bmp" || img_path.extension() == ".BMP")
        {
            cv::Mat img = cv::imread(img_path.string());
            if (!img.empty())
            {
                images.push_back(img);  // ���뵽��ƴ�ӵ�ͼ���б���
            }
            else
            {
                std::cout << "�޷���ȡͼƬ: " << img_path << std::endl;
            }
        }
    }

    if (images.size() < 2)
    {
        std::cout << "��Ҫ��������ͼƬ����ƴ�ӡ�" << std::endl;
        return -1;
    }

    // ʹ��SIFT�������
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // ���ڴ洢�ؼ����������
    std::vector<std::vector<cv::KeyPoint>> keypoints(images.size());
    std::vector<cv::Mat> descriptors(images.size());

    // ���ÿ��ͼƬ�Ĺؼ����������
    for (size_t i = 0; i < images.size(); ++i)
    {
        sift->detectAndCompute(images[i], cv::noArray(), keypoints[i], descriptors[i]);
    }

    // ʹ��BFMatcher����������ƥ��
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors[0], descriptors[1], matches);

    // ��ƥ������������
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
        });

    // ���ӻ�ƥ��㣬ֻ��ʾǰ10��ƥ���
    cv::Mat img_matches;
    cv::drawMatches(images[0], keypoints[0], images[1], keypoints[1], matches, img_matches, cv::Scalar::all(-1),
    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // ��ʾƥ���
    //cv::imshow("Matches", img_matches);
    cv::imwrite("matches.jpg", img_matches);  // ����ƥ�����
    cv::waitKey(0);  // �ȴ������˳�

    // ����Stitcherʵ��,ʹ�� SCANS ģʽ����ƴ�ӡ����в���Ϊ PANORAMA ��ƴ��ȫ��ͼ�������ٶȽ���������ƽ��ͼ���ƴ��Ч����̫��
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    // ������������ķֱ��ʱ�����Ӱ�����Ŷȣ���1��ʾȫ�ֱ��ʣ�ֵԽС�ٶ�Խ�쵫���ܽ��;���
    stitcher->setRegistrationResol(0.8);  // ����Ե������ֵ��������Ҫ�����ܺ�����������

    // ���ò���У��������ͼ����ʧ������
    stitcher->setWaveCorrection(false);

    // ƴ��ͼƬ
    cv::Mat pano;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Stitcher::Status status = stitcher->stitch(images, pano);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "ƴ��ͼƬ��ʱ: " << elapsed.count() << " ��" << std::endl;

    // ���ƴ��״̬
    if (status == cv::Stitcher::OK)
    {
        std::cout << "ƴ�ӳɹ���" << std::endl;
        fs::path output_path = img_folder / "stitched_output.bmp";
        cv::imwrite(output_path.string(), pano);  // ����ƴ�ӽ��

        // ���ƴ�ӽ���ķֱ���
        int width = pano.cols;
        int height = pano.rows;
        std::cout << "ƴ��ͼ��ķֱ���: " << width << "x" << height << std::endl;
    }
    else
    {
        std::cout << "ƴ��ʧ�ܣ��������: " << status << std::endl;
    }

    // �ȴ��û�����
    std::cout << "���س�������..." << std::endl;
    std::cin.get();  // �ȴ��û����»س���

    return 0;
}
