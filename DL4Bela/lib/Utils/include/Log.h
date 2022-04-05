#pragma once

#include <iostream>
#include <sstream>

namespace Utils
{
  class Log
  {
    std::stringstream stream_;

  public:
    explicit Log(const char *severity) { stream_ << severity << ": "; }
    std::stringstream &Stream() { return stream_; }
    ~Log() { std::cerr << stream_.str() << std::endl; }
  };

} // namespace Utils

#define LOG(severity) Utils::Log(#severity).Stream()
