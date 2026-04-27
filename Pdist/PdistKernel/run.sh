#!/bin/bash
ASCEND_RT_VISIBLE_DEVICES=0
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
# 默认 SOC 版本，可根据实际环境通过 -v 参数修改
SOC_VERSION="Ascend910B4"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

# 自动探测 Ascend Toolkit 安装路径
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
echo "Current compile soc version is ${SOC_VERSION}"

set -e
# 确保安装了 pybind11
pip3 install pybind11

# 清理并创建构建目录
rm -rf build
mkdir -p build

# 运行 CMake 配置
cmake -B build \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}

# 并行编译
cmake --build build -j

# 进入 build 目录运行测试脚本
(
    cd build
    # [修改点] 运行 pdist_custom 的测试脚本
    python3 ../pdist_custom_test.py
)