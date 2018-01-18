#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
#define M_PI 3.1415926

using namespace cv;
using namespace std;


static double rad2Deg(double rad) { return rad*(180 / M_PI); }//Convert radians to degrees
static double deg2Rad(double deg) { return deg*(M_PI / 180); }//Convert degrees to radians


void warpMatrix_annotation(Size   sz,
	double theta,
	double phi,
	double gamma,
	double scale,
	double fovy,
	Mat&   M,
	vector<Point2f>* corners) {
	double st = sin(deg2Rad(theta));
	double ct = cos(deg2Rad(theta));
	double sp = sin(deg2Rad(phi));
	double cp = cos(deg2Rad(phi));
	double sg = sin(deg2Rad(gamma));
	double cg = cos(deg2Rad(gamma));

	double halfFovy = fovy*0.5;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));
	double h = d / (2.0*sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);

	Mat F = Mat(4, 4, CV_64FC1);//Allocate 4x4 transformation matrix F
	Mat Rtheta = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around Z-axis by theta degrees
	Mat Rphi = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around X-axis by phi degrees
	Mat Rgamma = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around Y-axis by gamma degrees

	Mat T = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 translation matrix along Z-axis by -h units
	Mat P = Mat::zeros(4, 4, CV_64FC1);//Allocate 4x4 projection matrix

									   //Rtheta
	Rtheta.at<double>(0, 0) = Rtheta.at<double>(1, 1) = ct;
	Rtheta.at<double>(0, 1) = -st; Rtheta.at<double>(1, 0) = st;
	//Rphi
	Rphi.at<double>(1, 1) = Rphi.at<double>(2, 2) = cp;
	Rphi.at<double>(1, 2) = -sp; Rphi.at<double>(2, 1) = sp;
	//Rgamma
	Rgamma.at<double>(0, 0) = Rgamma.at<double>(2, 2) = cg;
	Rgamma.at<double>(0, 2) = sg; Rgamma.at<double>(2, 0) = sg;

	//T
	T.at<double>(2, 3) = -h;
	//P
	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0*f*n) / (f - n);
	P.at<double>(3, 2) = -1.0;
	//Compose transformations
	F = P*T*Rphi*Rtheta*Rgamma;//Matrix-multiply to produce master matrix

							   //Transform 4x4 points
	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW = sz.width / 2, halfH = sz.height / 2;
	Point2f annopt2f[4];
	annopt2f[0] = Point2f(0, 0);
	annopt2f[1] = Point2f(halfW / 2, 0);
	annopt2f[2] = Point2f(halfW / 2, halfH / 2);
	annopt2f[3] = Point2f(0, halfH / 2);

	for (int cnt = 0; cnt < 4; cnt++)
	{
		float cur_x = annopt2f[cnt].x;
		float cur_y = annopt2f[cnt].y;
		printf("%f, %f\n", cur_x, cur_y);
	}

	ptsIn[0] = annopt2f[0].x; ptsIn[1] = annopt2f[0].y;
	ptsIn[3] = annopt2f[1].x; ptsIn[4] = annopt2f[1].y;
	ptsIn[6] = annopt2f[2].x; ptsIn[7] = annopt2f[2].y;
	ptsIn[9] = annopt2f[3].x; ptsIn[10] = annopt2f[3].y;
	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0;//Set Z component to zero for all 4 components

	Mat ptsInMat(1, 4, CV_64FC3, ptsIn);
	Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);

	perspectiveTransform(ptsInMat, ptsOutMat, F);//Transform points

												 //Get 3x3 transform and warp image
	Point2f ptsInPt2f[4];
	Point2f ptsOutPt2f[4];

	for (int i = 0; i<4; i++) {
		Point2f ptIn(ptsIn[i * 3 + 0], ptsIn[i * 3 + 1]);
		Point2f ptOut(ptsOut[i * 3 + 0], ptsOut[i * 3 + 1]);
		ptsInPt2f[i] = ptIn + Point2f(halfW, halfH);
		ptsOutPt2f[i] = (ptOut + Point2f(1, 1))*(sideLength*0.5);
	}

	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	//Load corners vector
	if (corners) {
		corners->clear();
		corners->push_back(ptsOutPt2f[0]);//Push Top Left corner
		corners->push_back(ptsOutPt2f[1]);//Push Top Right corner
		corners->push_back(ptsOutPt2f[2]);//Push Bottom Right corner
		corners->push_back(ptsOutPt2f[3]);//Push Bottom Left corner
	}
}

void warpMatrix(Size   sz,
	double theta,
	double phi,
	double gamma,
	double scale,
	double fovy,
	Mat&   M,
	vector<Point2f>* corners) {
	double st = sin(deg2Rad(theta));
	double ct = cos(deg2Rad(theta));
	double sp = sin(deg2Rad(phi));
	double cp = cos(deg2Rad(phi));
	double sg = sin(deg2Rad(gamma));
	double cg = cos(deg2Rad(gamma));

	double halfFovy = fovy*0.5;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));
	double h = d / (2.0*sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);

	Mat F = Mat(4, 4, CV_64FC1);//Allocate 4x4 transformation matrix F
	Mat Rtheta = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around Z-axis by theta degrees
	Mat Rphi = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around X-axis by phi degrees
	Mat Rgamma = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 rotation matrix around Y-axis by gamma degrees

	Mat T = Mat::eye(4, 4, CV_64FC1);//Allocate 4x4 translation matrix along Z-axis by -h units
	Mat P = Mat::zeros(4, 4, CV_64FC1);//Allocate 4x4 projection matrix

									   //Rtheta
	Rtheta.at<double>(0, 0) = Rtheta.at<double>(1, 1) = ct;
	Rtheta.at<double>(0, 1) = -st; Rtheta.at<double>(1, 0) = st;
	//Rphi
	Rphi.at<double>(1, 1) = Rphi.at<double>(2, 2) = cp;
	Rphi.at<double>(1, 2) = -sp; Rphi.at<double>(2, 1) = sp;
	//Rgamma
	Rgamma.at<double>(0, 0) = Rgamma.at<double>(2, 2) = cg;
	Rgamma.at<double>(0, 2) = sg; Rgamma.at<double>(2, 0) = sg;

	//T
	T.at<double>(2, 3) = -h;
	//P
	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0*f*n) / (f - n);
	P.at<double>(3, 2) = -1.0;
	//Compose transformations
	F = P*T*Rphi*Rtheta*Rgamma;//Matrix-multiply to produce master matrix

							   //Transform 4x4 points
	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW = sz.width / 2, halfH = sz.height / 2;

	ptsIn[0] = -halfW; ptsIn[1] = halfH;
	ptsIn[3] = halfW; ptsIn[4] = halfH;
	ptsIn[6] = halfW; ptsIn[7] = -halfH;
	ptsIn[9] = -halfW; ptsIn[10] = -halfH;
	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0;//Set Z component to zero for all 4 components

	Mat ptsInMat(1, 4, CV_64FC3, ptsIn);
	Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);

	perspectiveTransform(ptsInMat, ptsOutMat, F);//Transform points

												 //Get 3x3 transform and warp image
	Point2f ptsInPt2f[4];
	Point2f ptsOutPt2f[4];

	for (int i = 0; i<4; i++) {
		Point2f ptIn(ptsIn[i * 3 + 0], ptsIn[i * 3 + 1]);
		Point2f ptOut(ptsOut[i * 3 + 0], ptsOut[i * 3 + 1]);
		ptsInPt2f[i] = ptIn + Point2f(halfW, halfH);
		ptsOutPt2f[i] = (ptOut + Point2f(1, 1))*(sideLength*0.5);
	}

	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	//Load corners vector
	if (corners) {
		corners->clear();
		corners->push_back(ptsOutPt2f[0]);//Push Top Left corner
		corners->push_back(ptsOutPt2f[1]);//Push Top Right corner
		corners->push_back(ptsOutPt2f[2]);//Push Bottom Right corner
		corners->push_back(ptsOutPt2f[3]);//Push Bottom Left corner
	}
}

void warpImage(const Mat &src,
	double    theta,
	double    phi,
	double    gamma,
	double    scale,
	double    fovy,
	Mat&      dst,
	Mat&      M,
	vector<Point2f> &corners,
	vector<Point2f> &annopts) {
	double halfFovy = fovy*0.5;
	double d = hypot(src.cols, src.rows);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));
	Mat im_temp = dst.clone();
	//imshow("dst_ori", dst);
	//waitKey(15);
	Point2f annopt2f[4];
	annopt2f[0] = Point2f(0, 0);
	annopt2f[1] = Point2f(src.cols / 4, 0);
	annopt2f[2] = Point2f(src.cols / 4, src.rows / 4);
	annopt2f[3] = Point2f(0, src.rows / 4);

	printf("warpMatrix_annotation running...\n");
	warpMatrix_annotation(src.size(), theta, phi, gamma, scale, fovy, M, &annopts);//Compute new annotation data
	for (int cnt = 0; cnt < annopts.size(); cnt++)
	{
		float cur_x = annopts[cnt].x;
		float cur_y = annopts[cnt].y;
		printf("%f, %f\n", cur_x, cur_y);
	}

	warpMatrix(src.size(), theta, phi, gamma, scale, fovy, M, &corners);//Compute warp matrix

	//warpPerspective(src, im_temp, M, im_temp.size(), INTER_NEAREST, BORDER_TRANSPARENT);// BORDER_TRANSPARENT);// BORDER_REPLICATE);// Size(sideLength, sideLength));//Do actual image warp
	warpPerspective(src, im_temp, M, im_temp.size(), INTER_NEAREST, BORDER_CONSTANT, (0));// BORDER_TRANSPARENT);// BORDER_REPLICATE);// Size(sideLength, sideLength));//Do actual image warp

	Point pts_dst[4];
	for (int i = 0; i < 4; i++)
	{
		pts_dst[i] = corners[i];
	}
	//imshow("dst_before_fill", dst);
	//waitKey(15);
	// Black out polygonal area in destination image.
	fillConvexPoly(dst, pts_dst, 4, Scalar(0), LINE_AA);
	//imshow("im_temp", im_temp);
	//imshow("dst", dst);
	//waitKey(15);
	dst = dst + im_temp;
}


int main(void) {
	Mat m, disp, warp, alter_cap;
	vector<Point2f> corners;
	vector<Point2f> annopts;
	VideoCapture capture; //声明视频读入类
	capture.open(0); //从摄像头读入视频 0表示从摄像头读入
	disp = imread("img_1.jpg");
	if (!capture.isOpened()) //先判断是否打开摄像头
	{
		cout << "can not open";
		cin.get();
		return -1;
	}
	string name = "Disp";
	namedWindow(name);
	string ori_img = "ori_img";
	namedWindow(ori_img);
	int i = 0;
	while (i == 0) {
		Mat cap; //定义一个Mat变量，用于存储每一帧的图像
		capture >> cap; //读取当前帧
		if (!cap.empty()) { //判断当前帧是否捕捉成功 **这步很重要
			imshow(ori_img, cap);
			//imwrite("cap.jpg", cap);
			waitKey(15);
			alter_cap = cap.clone();
			//imshow("bg_ori", disp);
			//waitKey(15);
			warpImage(alter_cap, 15, 15, 10, 0.75, 30, disp, warp, corners, annopts);
			imshow(name, disp);
			waitKey(15);
			imwrite("disp.jpg", disp);
			/*for (int cnt = 0; cnt < annopts.size(); cnt++)
			{
			float cur_x = annopts[cnt].x;
			float cur_y = annopts[cnt].y;
			printf("%f, %f\n", cur_x, cur_y);
			}*/

			//waitKey(15);
			i++;
		} //若当前帧捕捉成功，显示
		else
			cout << "can not ";
		//waitKey(30); //延时30毫秒
	}

	return 0;
	//warpImage(m, 5, 50, 0, 1, 30, disp, warp, corners);

}