#!/bin/bash
wget https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz
tar -xvf oidn-2.3.3.x86_64.linux.tar.gz
cd oidn-2.3.3.x86_64.linux
cp lib/libOpenImageDenoise_core.so.2.3.3 $CONDA_PREFIX/lib/python3.10/site-packages/sapien/oidn_library/
cp lib/libOpenImageDenoise_device_cuda.so.2.3.3 $CONDA_PREFIX/lib/python3.10/site-packages/sapien/oidn_library/
cp lib/libOpenImageDenoise.so.2.3.3 $CONDA_PREFIX/lib/python3.10/site-packages/sapien/oidn_library/
sed -i 's/2\.0\.1/2.3.3/g' $CONDA_PREFIX/lib/python3.10/site-packages/sapien/_oidn_tricks.py
