
<div align="center">

# Classification: 4 Monitor Lizard Species in Thailand  🦎
➳ [Presentation](https://www.canva.com/design/DAGJgcWFU_o/FKQlKFO2vxaEVLSNRKEeGw/view)
&nbsp;&nbsp;&nbsp;
➳ [Member & Responsibility](https://github.com/bubblebolt/deep-learning/blob/b745e7cd25a62d285c456ecdb688d128ccac3f0a/HW-CNN/HW_%E0%B9%80%E0%B8%9B%E0%B9%87%E0%B8%99...%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B8%84%E0%B8%B0_member.pdf)
&nbsp;&nbsp;&nbsp;
➳ [Download Dataset](https://github.com/bubblebolt/deep-learning/blob/ca0f6f7fee237002a6f50eed5b10242ac5b45cd8/HW-CNN/dataset.zip)

&nbsp;&nbsp;&nbsp;

<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/type_moniter_lizard.png" width="1000"> 


&nbsp;&nbsp;&nbsp;

</div>


## Introduction 💬
Develop an image classification model using a Convolutional Neural Network (CNN) to accurately identify and distinguish between different species of monitor lizards found in Thailand.

## Dataset Description 📦
### Source and Collection Method:
- This dataset contains images of monitor lizards from Thailand, focusing on four specific species. Images were gathered using the "Download All Images" extension in Google Chrome, primarily sourced from Google image searches and the iNaturalist platform.

### Composition:
- **Total Images:** 400
- **Classes:** 4 (each corresponding to a monitor lizard species)
- **Images per Class:** 100

### Dataset Balance:
- The dataset is evenly balanced with 100 images per class, ensuring unbiased model training and evaluation.

### Monitor Lizard Species:
1. Water Monitor (Varanus salvator)
2. Clouded Monitor (Varanus nebulosus)
3. Roughneck Monitor (Varanus rudicollis)
4. Dumeril's Monitor (Varanus dumerilii)



## Data Preparation 🧼
### Image Collection
- Images were collected from Internet sources.
- Unnecessary images were manually removed.
- Images were saved in `.jpg` or `.jpeg` format.
- Rescaled images to 224x224 pixels.

### Data Splitting
- `random_state = ['1234', '567', '8910']`
- `test_size = 0.2`
- `validation_split = 0.2`
- `stratify = y`

### Dataset Distribution
- **Train set:** 256 Pics 
- **Validation set:** 64 Pics
- **Test set:** 80 Pics

### Data Augmentation
Used Image augmentation layers in Keras to increase data diversity.
Applied flip, rotation, zoom, and contrast techniques.
```python
import tensorflow as tf

dataaug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1)
])
```


## Model Architecture⚒️

In this experiment, we will compare the performance of three pre-trained CNN models:
**VGG16**, **ResNet50**, **EfficientNetV2B0**

The fine-tuning process was tailored to each pre-trained model, considering their unique architectures and characteristics.

<div align="center">
<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/model_architecture.png" width="700"> 
</div>

## Training method 📈
Each pre-trained model was fine-tuned extensively, with numerous iterations of hyperparameter adjustments to optimize performance. We prioritized achieving the highest accuracy, minimizing loss, and avoiding overfitting.
<div align="center">
<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/parameters.png" width="700"> 
</div>


## Experiment results 📊
<div align="center">
  
<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/results.png" width="700"> 
<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/compare.png" width="700"> 

</div>
<br><br>

- `EfficientNetV2B0`🏆 shows superior performance in both training and validation accuracy compared to `VGG16` and `ResNet50`.
- `EfficientNetV2B0` emerges as the top performer after fine-tuning, achieving the highest accuracy, precision, recall, and F1-score among the three models.
- The results demonstrate that fine-tuning significantly enhances the performance of pre-trained models on this specific task of classifying Thai monitor lizard species.

<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/Grad-CAM.png" width="1000"> 

## Discussion : Grad-Cam checking 🔍
- Scoring system for evaluating Grad-CAM results based on correct predictions of 64 images.

<div align="center">
    
| Score | Description                             | Class No. of Pics | Percent   |
|-------|-----------------------------------------|-------------------|-----------|
| 0     | No relevance to lizard detected         | 23                | 35.94%    |
| 1     | Partial detection of lizard            | 14                | 21.88%    |
| 2     | Full detection of the entire lizard    | 27                | 42.19%    |
| Total |                                         | 64                | 100.00%   |

<img src="https://raw.githubusercontent.com/bubblebolt/deep-learning/main/HW-CNN/Pics/scoring.png" width="1000"> 

</div>



- **The confusion matrix indicates high values, suggesting some confusion among classes. However, Grad-CAM analysis reveals that approximately 35% of the highlighted areas are unrelated to the correct classification, despite the high scores in the confusion matrix.**
  


## Collaborators 🤝🏻

| Name          | Student ID  |
|---------------|-------------|
| Chalita Iamleelaporn | 6610412002 |
| Ranakorn Boonsuankergchai| 6610412003 |
| Tanwalai Yoongkieo | 6610422010 |
| Nattawut Intanai| 6610422023 |
| Patcharaporn Tuntino | 6610422030 |

For further details and contributions, refer to the project repository.

