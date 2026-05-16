#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="$(nproc)"

prepare_build_dir() {
    local build_dir="$1"
    local source_dir="$2"
    local cache_file="${build_dir}/CMakeCache.txt"

    mkdir -p "${build_dir}"

    if [ -f "${cache_file}" ] && ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=${source_dir}" "${cache_file}"; then
        echo "Removing stale CMake cache in ${build_dir} ..."
        rm -rf "${cache_file}" "${build_dir}/CMakeFiles"
    fi
}

build_cmake_project() {
    local source_dir="$1"
    local build_dir="${source_dir}/build"

    prepare_build_dir "${build_dir}" "${source_dir}"
    cd "${build_dir}"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"${BUILD_JOBS}"
}

echo "Configuring and building Thirdparty/DBoW2 ..."
build_cmake_project "${ROOT_DIR}/Thirdparty/DBoW2"

echo "Configuring and building Thirdparty/g2o ..."
build_cmake_project "${ROOT_DIR}/Thirdparty/g2o"

echo "Configuring and building Thirdparty/Sophus ..."
build_cmake_project "${ROOT_DIR}/Thirdparty/Sophus"

echo "Uncompress vocabulary ..."
cd "${ROOT_DIR}/Vocabulary"
tar -xf ORBvoc.txt.tar.gz

echo "Configuring and building DIF-SLAM ..."
build_cmake_project "${ROOT_DIR}"
