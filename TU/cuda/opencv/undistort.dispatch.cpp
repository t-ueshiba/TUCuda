#include "precomp.hpp"
#include "distortion_model.hpp"

#include "calib3d_c_api.h"

#include "undistort.simd.hpp"
#include "undistort.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv
{

void initInverseRectificationMap( InputArray _cameraMatrix, InputArray _distCoeffs,
                              InputArray _matR, InputArray _newCameraMatrix,
                              const Size& size, int m1type, OutputArray _map1, OutputArray _map2 )
{
    // Parameters
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();
    Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    // Check m1type validity
    if( m1type <= 0 )
        m1type = CV_16SC2;
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );

    // Init Maps
    _map1.create( size, m1type );
    Mat map1 = _map1.getMat(), map2;
    if( m1type != CV_32FC2 )
    {
        _map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        map2 = _map2.getMat();
    }
    else {
        _map2.release();
    }

    // Init camera intrinsics
    Mat_<double> A = Mat_<double>(cameraMatrix), Ar;
    if( !newCameraMatrix.empty() )
        Ar = Mat_<double>(newCameraMatrix);
    else
        Ar = getDefaultNewCameraMatrix( A, size, true );
    CV_Assert( A.size() == Size(3,3) );
    CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));

    // Init rotation matrix
    Mat_<double> R = Mat_<double>::eye(3, 3);
    if( !matR.empty() )
    {
        R = Mat_<double>(matR);
        //Note, do not inverse
    }
    CV_Assert( Size(3,3) == R.size() );

    // Init distortion vector
    if( !distCoeffs.empty() ){
        distCoeffs = Mat_<double>(distCoeffs);

        // Fix distortion vector orientation
        if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() ) {
            distCoeffs = distCoeffs.t();
        }
    }

    // Validate distortion vector size
    CV_Assert(  distCoeffs.empty() || // Empty allows cv::undistortPoints to skip distortion
                distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
                distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
                distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1) ||
                distCoeffs.size() == Size(1, 12) || distCoeffs.size() == Size(12, 1) ||
                distCoeffs.size() == Size(1, 14) || distCoeffs.size() == Size(14, 1));

    // Create objectPoints
    std::vector<cv::Point2i> p2i_objPoints;
    std::vector<cv::Point2f> p2f_objPoints;
    for (int r = 0; r < size.height; r++)
    {
        for (int c = 0; c < size.width; c++)
        {
            p2i_objPoints.push_back(cv::Point2i(c, r));
            p2f_objPoints.push_back(cv::Point2f(static_cast<float>(c), static_cast<float>(r)));
        }
    }

    // Undistort
    std::vector<cv::Point2f> p2f_objPoints_undistorted;
    undistortPoints(
        p2f_objPoints,
        p2f_objPoints_undistorted,
        A,
        distCoeffs,
        cv::Mat::eye(cv::Size(3, 3), CV_64FC1), // R
        cv::Mat::eye(cv::Size(3, 3), CV_64FC1) // P = New K
    );

    // Rectify
    std::vector<cv::Point2f> p2f_sourcePoints_pinHole;
    perspectiveTransform(
        p2f_objPoints_undistorted,
        p2f_sourcePoints_pinHole,
        R
    );

    // Project points back to camera coordinates.
    std::vector<cv::Point2f> p2f_sourcePoints;
    undistortPoints(
        p2f_sourcePoints_pinHole,
        p2f_sourcePoints,
        cv::Mat::eye(cv::Size(3, 3), CV_32FC1), // K
        cv::Mat::zeros(cv::Size(1, 4), CV_32FC1), // Distortion
        cv::Mat::eye(cv::Size(3, 3), CV_32FC1), // R
        Ar // New K
    );

    // Copy to map
    if (m1type == CV_16SC2) {
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<Vec2s>(p2i_objPoints[i].y, p2i_objPoints[i].x) = Vec2s(saturate_cast<short>(p2f_sourcePoints[i].x), saturate_cast<short>(p2f_sourcePoints[i].y));
        }
    } else if (m1type == CV_32FC2) {
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<Vec2f>(p2i_objPoints[i].y, p2i_objPoints[i].x) = Vec2f(p2f_sourcePoints[i]);
        }
    } else { // m1type == CV_32FC1
        for (size_t i=0; i < p2i_objPoints.size(); i++) {
            map1.at<float>(p2i_objPoints[i].y, p2i_objPoints[i].x) = p2f_sourcePoints[i].x;
            map2.at<float>(p2i_objPoints[i].y, p2i_objPoints[i].x) = p2f_sourcePoints[i].y;
        }
    }
}

void undistort( InputArray _src, OutputArray _dst, InputArray _cameraMatrix,
                InputArray _distCoeffs, InputArray _newCameraMatrix )
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    CV_Assert( dst.data != src.data );

    int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
    Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

    Mat_<double> A, Ar, I = Mat_<double>::eye(3,3);

    cameraMatrix.convertTo(A, CV_64F);
    if( !distCoeffs.empty() )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(5, 1, CV_64F);
        distCoeffs = 0.;
    }

    if( !newCameraMatrix.empty() )
        newCameraMatrix.convertTo(Ar, CV_64F);
    else
        A.copyTo(Ar);

    double v0 = Ar(1, 2);
    for( int y = 0; y < src.rows; y += stripe_size0 )
    {
        int stripe_size = std::min( stripe_size0, src.rows - y );
        Ar(1, 2) = v0 - y;
        Mat map1_part = map1.rowRange(0, stripe_size),
            map2_part = map2.rowRange(0, stripe_size),
            dst_part = dst.rowRange(y, y + stripe_size);

        initUndistortRectifyMap( A, distCoeffs, I, Ar, Size(src.cols, stripe_size),
                                 map1_part.type(), map1_part, map2_part );
        remap( src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT );
    }
}

}

CV_IMPL void
cvUndistort2( const CvArr* srcarr, CvArr* dstarr, const CvMat* Aarr, const CvMat* dist_coeffs, const CvMat* newAarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs = cv::cvarrToMat(dist_coeffs), newA;
    if( newAarr )
        newA = cv::cvarrToMat(newAarr);

    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
    cv::undistort( src, dst, A, distCoeffs, newA );
}


CV_IMPL void cvInitUndistortMap( const CvMat* Aarr, const CvMat* dist_coeffs,
                                 CvArr* mapxarr, CvArr* mapyarr )
{
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs = cv::cvarrToMat(dist_coeffs);
    cv::Mat mapx = cv::cvarrToMat(mapxarr), mapy, mapx0 = mapx, mapy0;

    if( mapyarr )
        mapy0 = mapy = cv::cvarrToMat(mapyarr);

    cv::initUndistortRectifyMap( A, distCoeffs, cv::Mat(), A,
                                 mapx.size(), mapx.type(), mapx, mapy );
    CV_Assert( mapx0.data == mapx.data && mapy0.data == mapy.data );
}

void
cvInitUndistortRectifyMap( const CvMat* Aarr, const CvMat* dist_coeffs,
    const CvMat *Rarr, const CvMat* ArArr, CvArr* mapxarr, CvArr* mapyarr )
{
    cv::Mat A = cv::cvarrToMat(Aarr), distCoeffs, R, Ar;
    cv::Mat mapx = cv::cvarrToMat(mapxarr), mapy, mapx0 = mapx, mapy0;

    if( mapyarr )
        mapy0 = mapy = cv::cvarrToMat(mapyarr);

    if( dist_coeffs )
        distCoeffs = cv::cvarrToMat(dist_coeffs);
    if( Rarr )
        R = cv::cvarrToMat(Rarr);
    if( ArArr )
        Ar = cv::cvarrToMat(ArArr);

    cv::initUndistortRectifyMap( A, distCoeffs, R, Ar, mapx.size(), mapx.type(), mapx, mapy );
    CV_Assert( mapx0.data == mapx.data && mapy0.data == mapy.data );
}

static void
cvUndistortPointsInternal(const CvMat* _src, CvMat* _dst,
			  const CvMat* _cameraMatrix, const CvMat* _distCoeffs,
			  const CvMat* matR, const CvMat* matP,
			  cv::TermCriteria criteria)
{
    double	A[3][3], RR[3][3], k[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    CvMat	matA=cvMat(3, 3, CV_64F, A), _Dk;
    CvMat	_RR=cvMat(3, 3, CV_64F, RR);
    cv::Matx33d	invMatTilt = cv::Matx33d::eye();
    cv::Matx33d	matTilt = cv::Matx33d::eye();

    cvConvert(_cameraMatrix, &matA);


    if( _distCoeffs )
    {
        _Dk = cvMat(_distCoeffs->rows, _distCoeffs->cols,
		    CV_MAKETYPE(CV_64F,CV_MAT_CN(_distCoeffs->type)), k);
        cvConvert(_distCoeffs, &_Dk);
    }

    if(matR)
        cvConvert(matR, &_RR);
    else
        cvSetIdentity(&_RR);

    if(matP)
    {
        double PP[3][3];
        CvMat _P3x3, _PP=cvMat(3, 3, CV_64F, PP);
        cvConvert( cvGetCols(matP, &_P3x3, 0, 3), &_PP );
        cvMatMul(&_PP, &_RR, &_RR);
    }

    const CvPoint2D32f*	srcf = (const CvPoint2D32f*)_src->data.ptr;
    const CvPoint2D64f*	srcd = (const CvPoint2D64f*)_src->data.ptr;
    CvPoint2D32f*	dstf = (CvPoint2D32f*)_dst->data.ptr;
    CvPoint2D64f*	dstd = (CvPoint2D64f*)_dst->data.ptr;
    int			stype = CV_MAT_TYPE(_src->type);
    int			dtype = CV_MAT_TYPE(_dst->type);
    int			sstep = (_src->rows == 1 ?
				 1 : _src->step/CV_ELEM_SIZE(stype));
    int			dstep = (_dst->rows == 1 ?
				 1 : _dst->step/CV_ELEM_SIZE(dtype));

    double		fx = A[0][0];
    double		fy = A[1][1];
    double		ifx = 1./fx;
    double		ify = 1./fy;
    double		cx = A[0][2];
    double		cy = A[1][2];

    int n = _src->rows + _src->cols - 1;
    for( int i = 0; i < n; i++ )
    {
        double x, y, x0 = 0, y0 = 0, u, v;
        if( stype == CV_32FC2 )
        {
            x = srcf[i*sstep].x;
            y = srcf[i*sstep].y;
        }
        else
        {
            x = srcd[i*sstep].x;
            y = srcd[i*sstep].y;
        }
        u = x;
	v = y;
        x = (x - cx)*ifx;
        y = (y - cy)*ify;

        if( _distCoeffs )
	{
	  // compensate tilt distortion
            cv::Vec3d	vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);
            double	invProj   = vecUntilt(2) ? 1./vecUntilt(2) : 1;
            x0 = x = invProj * vecUntilt(0);
            y0 = y = invProj * vecUntilt(1);

            double	error = std::numeric_limits<double>::max();

	  // compensate distortion iteratively
            for( int j = 0; ; j++)
            {
                if ((criteria.type & cv::TermCriteria::COUNT) &&
		    j >= criteria.maxCount)
                    break;
                if ((criteria.type & cv::TermCriteria::EPS) &&
		    error < criteria.epsilon)
                    break;
                double	r2 = x*x + y*y;
                double	icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)
			       / (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
                if (icdist < 0)  // test: undistortPoints.regression_14583
                {
                    x = (u - cx)*ifx;
                    y = (v - cy)*ify;
                    break;
                }
                double	deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)
			       + k[8]*r2+k[9]*r2*r2;
                double	deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y
			       + k[10]*r2+k[11]*r2*r2;
                x = (x0 - deltaX)*icdist;
                y = (y0 - deltaY)*icdist;

                if (criteria.type & cv::TermCriteria::EPS)
                {
                    double r4, r6, a1, a2, a3, cdist, icdist2;
                    double xd, yd, xd0, yd0;
                    cv::Vec3d vecTilt;

                    r2 = x*x + y*y;
                    r4 = r2*r2;
                    r6 = r4*r2;
                    a1 = 2*x*y;
                    a2 = r2 + 2*x*x;
                    a3 = r2 + 2*y*y;
                    cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
                    icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
                    xd0 = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
                    yd0 = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

                    vecTilt = matTilt*cv::Vec3d(xd0, yd0, 1);
                    invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
                    xd = invProj * vecTilt(0);
                    yd = invProj * vecTilt(1);

                    double x_proj = xd*fx + cx;
                    double y_proj = yd*fy + cy;

                    error = sqrt( pow(x_proj - u, 2) + pow(y_proj - v, 2) );
                }
            }
        }

        double	xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
        double	yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
        double	ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
        x = xx*ww;
        y = yy*ww;

        if( dtype == CV_32FC2 )
        {
            dstf[i*dstep].x = (float)x;
            dstf[i*dstep].y = (float)y;
        }
        else
        {
            dstd[i*dstep].x = x;
            dstd[i*dstep].y = y;
        }
    }
}

} // namespace
/*  End of file  */
