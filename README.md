# Face Forgery Detection with CLIP-Enhanced Multi-Encoder Distillation

This repository contains the code for the paper Face Forgery Detection with CLIP-Enhanced Multi-Encoder Distillation 


### Abstract

With the development of face forgery technology, fake faces are rampant, threatening the security and authenticity of many fields. Therefore, it is of great significance to study face forgery detection. At present, existing detection methods have deficiencies in the comprehensiveness of feature extraction and model adaptability, and it is difficult to accurately deal with complex and changeable forgery scenarios. However, the rise of multimodal models provides new insights for current forgery detection methods. At present, most methods use relatively simple text prompts to describe the difference between real and fake faces. However, these researchers ignore that the CLIP model itself does not have the relevant knowledge of forgery detection. Therefore, our paper proposes a face forgery detection method based on multi-encoder fusion and cross-modal knowledge distillation. On the one hand, the prior knowledge of the CLIP model and the forgery model is fused. On the other hand, through the alignment distillation, the student model can learn the visual abnormal patterns and semantic features of the forged samples captured by the teacher model. Specifically, our paper extracts the features of face photos by fusing the CLIP text encoder and the CLIP image encoder, and uses the dataset in the field of forgery detection to pretrain and fine-tune the Deepfake-V2-Model to enhance the detection ability, which are regarded as the teacher model. At the same time, the visual and language patterns of the teacher model are aligned with the visual patterns of the pretrained student model, and the aligned representations are refined to the student model. This not only combines the rich representation of the CLIP image encoder and the excellent generalization ability of text embedding, but also enables the original model to effectively acquire relevant knowledge for forgery detection. Experiments show that our method effectively improves the performance on face forgery detection.

## Code

### Installing dependencies

```sh
pip install -r requirements.txt
```


## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.


```
bash scripts/<stud_model>_student/run_stage1.sh <gpu_id> <dataset_id> <dataset_path>
```

```

Here, ```<stud_model>``` refers to the student architecture, which can be one of the following: rn50, vitb, vits, deits. ```<gpu_id>``` is the GPU ID of the machine. ```<dataset_id>``` is the dataset . ```<dataset_path>``` is the parent folder of all the datasets given with the "/" included. (Eg: /my_folder/datasets/) 


our checkpoint can be downloaded at https://pan.baidu.com/s/1H20B5pRrtFRtzG-_YFg0Uw
