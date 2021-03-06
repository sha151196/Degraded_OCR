# OCR for Degraded Bangla Documents
A CNN BLSTM CTC based implementation of Bangla Degraded OCR line recognition.
## Features
+ Takes Grayscale / Binarized line image
+ No Feature Extraction is required
+ Generates Encoded groundtruth file for each of the line image directory.
## Requirements
This model is implemented using
1. Python 2.7 (maintain this version)
2. Tensorflow 1.6+
3. Pillow
4. Numpy
5. Scipy
## Usage Instruction
* Run the **GT_Encode.py** file, to create an encoded groundtruth file for each of image folder, as per the given **mode**, i.e. **Train** and **Test**. Run this for **Train** and **Test** directory seperately. A encoded groundtruth text file will be created for both of the aforementioned directories.
* A CNN BLSTM CTC based network is implemented as in Figure:
![Model][model]

[model]: https://github.com/sha151196/Degraded_OCR/blob/master/Inception-ctc.jpg "Architecture"
* The network is given in **Hybrid_Model_Degraded.py**. Run the ```main()``` method as specified in comments.
* In ```Predict``` mode the network will genrate a file (**Predicted.txt**) containing actual annotation and network predicted strings.
