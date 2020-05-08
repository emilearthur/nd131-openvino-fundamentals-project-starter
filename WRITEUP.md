# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom Layers are layers that are not in the list of known layers.
The model optimizer compares each layer of a model with a known layer list, If model contains any layer not listed then it is been classified as a custom layer.

The process behind converting custom layers depends on the framework one is using, either tensorflow, Caffe, Kaldi, MXNet or ONYX.  
For more visit: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

The process behind converting custom layers involves...

* Generating the Extension Template files using the Model Extension Generator (MOG).
* Using the Model Optimizer to Generate IR Files containing the Custom Layer.
* Edit the CPU Extension Template files.
* Excute the Model with the Custom Layer.

Example Below:

* Download the SSD MobileNet V2 COCO model from Tensorflow

``` wget
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

```

* Use tar -xvf command to unpack the zipped model file.

``` tar
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

* Cd into the models directory and convert the Tensorflow model by feeding SSD MobileNet V2 COCO model's .pb file using the model optimizer.

``` MOG
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

* Successful conversion results to generation of a .xml and .bin IR model files.

Some of the potential reasons for handling custom layers are:

* To optimize pre-trained models and convert them to Intermediate Representation (IR) without loss of accuracy and increase in perfomance.

To run the project, use the following commands:
using video file

```video
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

using camera stream:

```stream
python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

```

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

* The inference time of the pre-conversion model ( [person_retail_0013_description_person_detection_retail_0013](https://docs.openvinotoolkit.org/2019_R3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)) was 0.047s and that of the post-conversion model (SSD MobileNet V2 COCO) was 0.72s.

The size of the model pre- and post-conversion was...

* The size of the post-coonversion model (SSD MobileNet V2 COCO), i.e. IR XML + BIN file is 64.22MB is larger than the size of the pre-conversion model (person_retail_0013_description_person_detection_retail_0013), i.e. XML + BIN file is 1565KB.

The inference time of the model pre- and post-conversion was...

* The inference time of the pre-conversion model (person_retail_0013_description_person_detection_retail_0013) was 0.047sec and it had less inference time compared to the post-coonversion model (SSD MobileNet V2 COCO) which was 0.74sec.

compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge...

* Edge models needs only a local network connection to function and they can also perform on low speed networks compared to cloud services which need high speed internet to function which is costly.

* Cost of renting a server (eg. AWS, GCP, Azure and etc) is high compared to running edge models on minimal cpu on local network connection.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

* The retail industry. This will enable them know the number of customers enter their stores and the section of the store the customers spend most of their time.
* Estimation of wildlife population.
* Automatic delivery system.

Each of these use cases would be useful because...

* It helps to know when to stock some section of the store (operation stratagies) and also in security control.
* It can help zoos and national parks avoid overpopulation of species.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

* Lighting: Lighting is an essential factor to the results of the model becuase increase/decrease has a direct impact on the number of false positives obtained. Thus the place monitored should have sufficient light to get good accuracy of the model.
* Model Accuracy: Since model is deployed on the edge and results is need in realtime, it should have high accuracy to help end users get the desired results. High/Low accuracy has impact on the number of false postives which affects statistical measurement.
* Camera focal Length:  High focal length results to good focus on object and narrow angle image whiles the other is vice versa. Depending on what the end-user wants, it should noted that each choice has an impact on dectected situation and it can make it better or worse. Eg. A user who wants monitor a wider place than have more focus will end up with a model which extracts less information. This lowers the accuracy of the model.
* Image Size: Image size is dependant on resolution. If image resolution is higher then the size is larger. Model gives better output on images with higher resolution but his takes time and space. Thus if end user has enough space and little delay does not cause an issue then higher resolution can be used else otherwise.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
