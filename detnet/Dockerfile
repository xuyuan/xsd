# docker build -t wine_detector .
# docker run --rm --gpus all --shm-size=8G --network none --name="wine-detector-run" -v /data:/tcdata wine_detector


FROM registry.cn-shanghai.aliyuncs.com/airobot_org/wine_detector:5.6

ENV LANG C.UTF-8

#RUN apt-get install -y libgl1-mesa-glx

#RUN curl --silent -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda
#RUN rm Miniconda3-latest-Linux-x86_64.sh
#ENV PATH="/root/miniconda/bin:$PATH"
#ENV PYTORCH_CHECKPOINTS='/root/.cache/torch/checkpoints'
#

ENV CODE_PATH="/bottle_defects_detection"
ADD . $CODE_PATH
#RUN rm -rf $CODE_PATH/.git*
#RUN rm -rf $CODE_PATH/Dockerfile
#RUN mkdir -p $PYTORCH_CHECKPOINTS
#RUN mv $CODE_PATH/data/efficientnet-b5-b6417697.pth $PYTORCH_CHECKPOINTS
#

#RUN mv /run.sh /run.sh.bak
#COPY run.sh /

#COPY test.sh /competition/code
#COPY train.sh /competition/code

#RUN mv /root/.condarc /root/cn.condarc
#RUN conda config --show-sources
#RUN conda env update

#WORKDIR $CODE_PATH
WORKDIR /

#ADD test.sh $CODE_PATH

#VOLUME ["/tcdata"]
#CMD ["sh", "run.sh"]
