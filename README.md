# Machine learning Project
## Nuclei detection

### Scope
this project explores the increased performance of the Unet architecture vs a benchmark network (made up of four convolutional layers). This project also analyzes the increased benefit of augmenting data.

### Install

This project requires **Python 3.5** and the following libraries installed:

- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [sys](https://www.scipy.org/)
- [os](https://docs.python.org/2/library/os.html)
- [random](https://docs.python.org/3/library/random.html)
- [warnings](https://docs.python.org/2/library/warnings.html)
- [time](https://docs.python.org/2/library/time.html)
- [matplotlib.pyplot](http://matplotlib.org/)
- [collections](https://docs.python.org/2/library/collections.html)
- [skimage.io](http://scikit-image.org/docs/dev/api/skimage.io.html)
- [skimage.transform](http://scikit-image.org/docs/dev/api/skimage.transform.html)
- [skimage.morphology](http://scikit-image.org/docs/dev/api/skimage.morphology.html)
- [keras.preprocessing.image](https://keras.io/preprocessing/image/)
- [keras.models](https://keras.io/models/about-keras-models/)
- [keras.layers](https://keras.io/layers/about-keras-layers/)
- [keras.layers.convolutional](https://keras.io/layers/convolutional/)
- [keras.layers.core](https://keras.io/layers/core/)
- [keras.layers.pooling](https://keras.io/layers/pooling/)
- [keras.layers.merge](https://keras.io/layers/merge/)
- [keras.callbacks](https://keras.io/callbacks/)
- [keras.backend](https://keras.io/backend/)
- [sklearn.model_selection](http://scikit-learn.org/stable/model_selection.html)
- [tensorflow](https://www.tensorflow.org/)
- [tqdm](https://tqdm.github.io/)
- [jupyter](http://ipython.org/notebook.html)
- [Anaconda](http://continuum.io/downloads)

### Code

A template notebook is provided as `notebook.ipynb`.

### Input

The [Kaggle competition webpage](https://www.kaggle.com/c/data-science-bowl-2018/data) provides the needed training data and a separate set of images for testing. The training data contains the original image and a series of masks. The testing data only contains the original test image. 

File Descriptions
- 	/stage1_train/* - training set images (images and annotated masks). The training set has 670 folders for 670 images in png format (one folder per image). Each folder has two sub-folders (masks and images). The sub-folder images has only one png file (the image) and the sub-folder masks contains as many files as nuclei.  

- /stage1_test/* - stage 1 test set images (images only). The set has 65 images in jpg format. The input folder contains the training and test datasets.


### Output

The ouput folder contains 3 models (.h5 files) and the corresponding 3 csv files.

### Run

The models were run on AWS. I created an instance using the following:
- image: [Deep Learning AMI with Source Code (CUDA 8, Ubuntu)](https://aws.amazon.com/marketplace/pp/B06VSPXKDX)
    This AMI comes with TensorFlow 1.3.0 and Keras 2.0.8 with TensorFlow as default backend.
- instance type: p2.xlarge
- Security Group:
 - Port range: 8888 (Custom TCP Rule, Source 0.0.0.0/0)
 - Port range: 22 (Custom TCP Rule, Source 0.0.0.0/0)
 - Port range: 8888 (ssh, Source 0.0.0.0/0) 

A PEM certificate is created when launching the instance.

Once the instance is created, follow these steps:
- Go back to the AWS console and copy the public DNS (public_DNS)

- Initiate the connection from the terminal into the notebook server:
    `ssh -i yourPemFile ubuntu@public_DNS` with yourPemFile being the PEM certificate.
- Create environment with the `conda create --name myenv` command where myenv is the name of the environment.
- Activate the environment with `source activate myenv`.
- Install anaconda with 

    `wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda3-4.0.0-Linux-x86_64.sh`

    `bash Anaconda3-4.0.0-Linux-x86_64.sh`
- Ensure the notebook uses the python module from the anaconda directory
    `source .bashrc`
- Check with `which python` the python directory
- Update jupyter with `conda update jupyter`
- Install pandas with `conda install pandas`
- Start up jupyter notebook with `jupyter notebook`
- Go to the browser and type in the following address: https//public_DNS:8888
- In the browser, click advanced and then "proceed to https//public_DNS:8888"
- In the browser enter the password:
- In the terminal, clone the github directory with `git clone yourGitHubFolderAddress .` where yourGitHubFolderAddress is the address of your repository. You can create a folder before that step and move into the folder before typing in the git clone command.

