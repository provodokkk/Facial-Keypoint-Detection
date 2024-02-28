# Facial Keypoint Detection

## About the Program
A system for identifying facial key points in an image using a Convolutional Neural Network (CNN).

## Content List
- [Project Structure](#anchor_1)
- [How to Setup the Project](#anchor_2)
- [How to Train the Model](#anchor_3)
- [How to Evaluate the Model](#anchor_4)
- [How to Add Custom Data to the Dataset](#anchor_5)
---

<a id = "anchor_1"></a>
### Project Structure
```
├── input
│   ├── test.csv
│   └── training.csv
├── outputs
│   ├── saved_model
│   │   ├── assets
│   │   ├── variables
│   │   │   ├── variables.data-00000-of-00001
│   │   │   └── variables.index
│   │   ├── keras_metadata.pb
│   │   └── saved_model.pb
│   ├── test_results
│   ├── validation_results
│   └── loss.png
├── src
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate_and_test.py
│   ├── model.py
│   ├── requirements.txt
│   ├── train.py
│   └── utils.py
├── haarcascade_frontalface_alt2.xml
```

<a id = "anchor_2"></a>
### How to Setup the Project

1. Clone the Project.
```
git clone git@github.com:provodokkk/Facial-Keypoint-Detection.git
```

2. Create a Virtual Environment inside the `Facial-Keypoint-Detection` directory.
```
python -m venv venv
```

3. Activate the Virtual Environment.
```
venv\Scripts\activate
```

4. Change Directory.
```
cd src
```

5. Install Requirements.
```
pip install -r requirements.txt
```

6. Create the `input` folder in the same directory as `src`.

7. Download the Dataset in the a folder.
The dataset used in this project is from [Kaggle Competition](https://www.kaggle.com/competitions/facial-keypoints-detection/data).

8. Extract the files as it shown int the Project Structure.


<a id = "anchor_3"></a>
### How to Train the Model
Run the following command in the terminal in the `src` directory.

```
python train.py 
```


<a id = "anchor_4"></a>
### How to Evaluate the Model
Run the following command in the terminal in the `src` directory.

```
python evaluate_and_test.py 
```

<a id = "anchor_5"></a>
### How to Add Custom Data to the Dataset
1. Place the image in the `src` directory.
2. Run the following command in the terminal in the `src` directory.
```
python image_processor.py
```
3. Add key points in the open window.\
`Points must be added in a certain order, hints can be found below the image.`

### Example
<p align="center">
  <img src="https://github.com/provodokkk/Facial-Keypoint-Detection/assets/105476685/ab8777c6-7a30-4da8-ae02-7fe65dc704b8"/>
</p>
