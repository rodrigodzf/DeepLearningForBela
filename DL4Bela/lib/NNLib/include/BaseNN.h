/**
  \ingroup NNLib
  \file    BaseNN
  \brief   This file contains the implementation for class BaseNN.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <vector>

class BaseNN
{

public:
    BaseNN(){};
    virtual ~BaseNN(){};

    virtual bool load(const std::string &filename) = 0;
    // virtual inline bool process(const std::vector<float> &inputData, std::vector<float> &outResults) = 0;
};
