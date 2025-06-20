FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Update package lists and install required packages
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --no-cache-dir \
    numpy\<2 \
    pandas \
    albumentations==1.2.1 \
    acvl-utils==0.2 \
    scikit-learn \
    tqdm \
    seaborn 

# Clone and configure nnUNet-for-pathology v1
RUN git clone https://github.com/DIAGNijmegen/nnUNet-for-pathology.git /home/user/nnunet && \
    git -C /home/user/nnunet checkout nnunet_for_pathology_v1 && \
    pip3 install -e /home/user/nnunet

# For (VS code) ssh connection 
RUN mkdir /var/run/sshd

ENTRYPOINT ["/bin/bash"]