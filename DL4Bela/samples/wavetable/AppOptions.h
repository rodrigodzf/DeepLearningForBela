#pragma once

#include <string>

struct AppOptions
{
    std::string modelPath = "";
    int frontend = 0;
    bool useArmnn = false;
    int wavetableSize = 1024;

};
