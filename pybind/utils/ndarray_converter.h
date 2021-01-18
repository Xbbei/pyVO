#pragma once

#include <Python.h>
#include <opencv2/core/core.hpp>

class NDArrayConverter {
public:
    // must call this first, or the other routines don't work!
    static bool init_numpy();
    
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};