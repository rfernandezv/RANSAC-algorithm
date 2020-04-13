#include<opencv/cv.h>
#include<opencv/highgui.h>
#include<iostream>
#include<io.h>
#include"ransac_line2d.h"
#include"ransac_circle2d.h"
#include"ransac_ellipse2d.h"

using namespace cv;
using namespace std;

bool testCircle2dMulti_rfv()
{
	cv::Mat src = cv::imread("images/test010.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat thresholdSrc;
	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	threshold(src, inImg, 90, 255, CV_THRESH_BINARY);
	
	thresholdSrc = inImg.clone();
	std::vector<sac::Point2D> pCloud2D;

	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	double specRadius = 17;  // 18

	sac::ransacModelCircle2D circle2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	circle2D.setDistanceThreshold(5);
	circle2D.setMaxIterations(2500);
	circle2D.setSpecificRadius(specRadius, 0.2);

	//imshow("threshold", src);
	if (pCloud2D.size() == 0) {
		cout << "pCloud2D.size() es igual a cero ";
		waitKey(5000);
	}

	int contador = 0;
	while (pCloud2D.size() > 500)
	{
		circle2D.setInputCloud(pCloud2D);
		circle2D.computeModel();
		circle2D.getInliers(inliers);
		circle2D.getModelCoefficients(parameter);

		if (inliers.size() < specRadius * 2 * CV_PI){
			cout << "Break inliers = 0. No circle detection";
			break;	
		}
		Point cp((int)parameter.modelParam[0], (int)parameter.modelParam[1]);
		int radius = (int)parameter.modelParam[2];
		circle(showMat, cp, radius, Scalar(0, 255, 0), 2, 8);
				
		imshow("circles", showMat);
		waitKey(100);

		cout << "Parameter of 2D line: < " << parameter.modelParam[0] << ", " <<
			parameter.modelParam[1] << " >---" << parameter.modelParam[2] << endl;

		circle2D.removeInliders(pCloud2D, inliers);
		contador++;

		/*if (contador > 8) {
			break;
		}*/
	}
	imwrite("threshold_detected.jpg", thresholdSrc);
	imshow("threshold", thresholdSrc);
	waitKey();

	cout << endl;

	return true;
}

bool testEllipse2dMulti_rfv()
{
	//cv::Mat src = cv::imread("ellipsesTest.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat src = cv::imread("images/test010.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat thresholdSrc;
	if (src.empty())
		return false;

	Mat inImg, showMat;
	cvtColor(src, showMat, CV_GRAY2BGR);
	//threshold(src, inImg, 10, 255, CV_THRESH_BINARY_INV);
	threshold(src, inImg, 75, 255, CV_THRESH_BINARY);
	thresholdSrc = inImg.clone();

	std::vector<sac::Point2D> pCloud2D;
	for (int i = 0; i < inImg.rows; i++)
	{
		uchar* p = inImg.ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			if (p[j] != 0)
				pCloud2D.push_back(sac::Point2D(j, i));
		}
	}

	sac::ransacModelEllipse2D ellipse2D;
	std::vector<int> inliers;
	sac::ModelCoefficient parameter;
	ellipse2D.setDistanceThreshold(5);
	ellipse2D.setMaxIterations(2500);
	//ellipse2D.setSpecficAxisLength(21, 12, 0.2);
	ellipse2D.setSpecficAxisLength(26, 9, 0.6);	// 35, 15

	while (pCloud2D.size() > 500)
	{
		cout << pCloud2D.size() << endl;
		ellipse2D.setInputCloud(pCloud2D);
		ellipse2D.computeModel();
		ellipse2D.getInliers(inliers);
		ellipse2D.getModelCoefficients(parameter);

		if (inliers.size() < 500)
			break;

		cv::Point2f ellipseCenter;
		ellipseCenter.x = (float)parameter.modelParam[0];
		ellipseCenter.y = (float)parameter.modelParam[1];
		cv::Size2f ellipseSize;
		ellipseSize.width = (float)parameter.modelParam[2] * 2;
		ellipseSize.height = (float)parameter.modelParam[3] * 2;

		float ellipseAngle = (float)parameter.modelParam[4];
		cout << "Parameters of ellipse2D: < " << parameter.modelParam[0] << ", " <<
			parameter.modelParam[1] << " > --- ";
		cout << "Long/Short Axis: " << parameter.modelParam[2] << "/" << parameter.modelParam[3] << " --- ";
		cout << "Angle: " << parameter.modelParam[4] << endl;

		cv::ellipse(showMat, cv::RotatedRect(ellipseCenter, ellipseSize, ellipseAngle), cv::Scalar(0, 255, 0), 2, 8);

		imshow("ellipses", showMat);
		waitKey(12);

		ellipse2D.removeInliders(pCloud2D, inliers);
	}
	imshow("threshold", thresholdSrc);
	cout << "Proceso terminado";

	waitKey();

	return true;
}

int main()
{
	testEllipse2dMulti_rfv();

	//testCircle2dMulti_rfv();	

	return 1;
}
