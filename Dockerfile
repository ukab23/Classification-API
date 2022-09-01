FROM condaforge/miniforge3

# Setting the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Getting the updates for Ubuntu and installing python into our environment
RUN pip install mlflow>=1.0 \
    && pip install azure-storage-blob==12.3.0 \
    && pip install numpy==1.21.2 \
    && pip install scipy \
    && pip install pandas==1.3.3 \
    && pip install scikit-learn==0.24.2 \
    && pip install cloudpickle

# RUN apt-get -y update  && apt-get install -y python

# Run api.py when the container launches
EXPOSE 8080
CMD ["python", "api.py"]