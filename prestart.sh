#!/bin/bash
# Remove problematic NVIDIA libraries
find /opt/render/project -name "libcufft.so*" -delete
find /opt/render/project -name "libcudnn*" -delete