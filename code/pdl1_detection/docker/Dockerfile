FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set & fill the working directory
WORKDIR /app
COPY . /app

# Install requirements + hacky fix for ensuring correct OpenCV version
RUN pip3 install -r /app/code/pdl1_detection/docker/requirements.txt
RUN pip3 install --force-reinstall opencv-python-headless
ENV PYTHONPATH="${PYTHONPATH}:/app/code/pdl1_detection"

ENTRYPOINT ["/bin/bash"]