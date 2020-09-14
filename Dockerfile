FROM codalab/default-gpu:v0.3.3

ARG CI_JOB_ID=''

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda/bin:$PATH"
#ENV PYTORCH_CHECKPOINTS='/root/.cache/torch/checkpoints'

ENV CODE_PATH="/object_cxr"
RUN mkdir $CODE_PATH
COPY environment.yml $CODE_PATH

WORKDIR $CODE_PATH

RUN conda env update
