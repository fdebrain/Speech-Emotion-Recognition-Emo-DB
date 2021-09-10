FROM python:3.7-slim
WORKDIR /home

# Install system dependencies
RUN apt update && \
    apt install -y wget unzip libgomp1 libsndfile1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Emo-DB dataset
RUN wget http://emodb.bilderbar.info/download/download.zip && \
    unzip download.zip -d data && \
    rm download.zip

# Copy modules and notebook
COPY modules modules/
COPY Report_SER.ipynb .
COPY app.py .

# Expose port to access Jupyter notebook
EXPOSE 8888

# Expose port to access Flask API
EXPOSE 8080

# Default command when container is run
CMD ["/bin/bash"]