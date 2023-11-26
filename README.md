## Data-Downloader

- on Linux, simply run `bash data_downloader.sh`
- on Windows, 
    - run `curl -O https://bitbucket.org/manishghop/fruits_360/get/f0a8c45800d3.zip`
    - unzip f0a8c45800d3.zip 
    - unzip manishghop-fruits_360-f0a8c45800d3/fruits.zip


## Requirements

Run `conda create -n mobile_classifier python==3.9`
`conda activate mobile_classifier`
`pip install -r requirements.txt`

## Training

- Run `python train.py` 

## Export

Run `python export_model_to_onnx.py`

Run `onnx-tf -i mobile_fruit_classification.onnx -o mobile_fruit_classification.pb`

Run `python export_onnx_tflite.py`
