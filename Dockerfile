FROM python:3.11-slim

WORKDIR /app

# 安装 OpenCV 和 MediaPipe 相关依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 库，包含 mediapipe
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    Flask==3.1.0 \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    requests==2.31.0 \
    Werkzeug==3.1.3 \
    mediapipe==0.10.21

# 复制项目代码
COPY . .

# 启动服务
CMD ["python", "app.py"]


# # 用依赖齐全的基础镜像
# FROM python:3.11

# # 设置工作目录
# WORKDIR /app

# # 安装系统依赖（opencv 和 mediapipe 需要）
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libegl1 \
#     && rm -rf /var/lib/apt/lists/*

# # 配置 pip 国内源
# # RUN mkdir -p /root/.pip && \
# #     echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /root/.pip/pip.conf
# RUN mkdir -p /root/.pip && \
#     echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /root/.pip/pip.conf

# # 先复制 requirements.txt
# COPY requirements.txt .

# # 安装依赖，强制只用 wheel，防止慢到爆的编译
# RUN pip install --only-binary=:all: --no-cache-dir -r requirements.txt || \
#     (echo "⚠️ 部分包没有 wheel，尝试普通安装" && pip install --no-cache-dir -r requirements.txt)

# # 再复制项目文件
# COPY . .

# # 启动命令
# CMD ["python", "app.py"]
