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

* Adding extensions to both the Model Optimizer and the Inference Engine.

Some of the potential reasons for handling custom layers are:

* To optimize pre-trained models and convert them to Intermediate Representation (IR) without loss of accuracy and increase in perfomacance.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

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

* Lighting: Lighting is an essential factor to the results of the model becuase increase/decrease has a direct impact on the number of false positives obtained. Thus the place monitored should have sufficient light.
* Model Accuracy: Model accuracy is important as it also direct effect in increase/decrease in false postives. Since model is deployed on the edge and results is need in realtime, it should have low high accuracy to help end users get the desired results. In all model accruracy affects statistical measurement.
* Camera focal Length:  High focal length results to good focus on object and narrow angle image whiles the other is vice versa. Depending on what the end-user want, it should noted that each choice either makes reaction of the dectected situation better or worse. Eg. A user who wants monitor a wider place than have more focus will end up with a model which extracts less information. This lowers the accuracy of the model.
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
