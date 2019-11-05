# OpenCV_Sidewalk_Detection
Sidewalk detection using python 3.7 and OpenCV

## Description
This is a basic python script meant to help detect sidewalks. The module is a part of a larger autonomous computer vision based drone project.

### Prerequisites

To deploy the program on a local machine the following hardware/softwares are requires :

#### Python 3.7
#### OpenCV
#### Numpy

To download required libraries, navigate to the project directory in terminal and type in the command below.

```
pip install -r requirements.txt
```

## Running the tests

You can add path to images in sidewalk.py and running the script will overwrite the image to produce red lines around the sidewalk.


## Problems

The algorithm seems fairly ineffective if used on low-resolution/dark images.
Currently running the script draws several lines around sidewalk instead of two lines.


## Authors

* **Hamza Ehsan** - *Autonomous Flight* - (www.hamzaehsan.com)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
