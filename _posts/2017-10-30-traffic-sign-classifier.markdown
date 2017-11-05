---
layout: post
title:  "CNN with Traffic Sign Classifier"
date:   2017-11-01 20:25:27 +0100
---

[//]: # (Image References)
[false_pos]: /assets/false_positive.png
[sliding]: /assets/sliding_windows.png
[works]: /assets/works.png
[output_8_0]: /assets/output_8_0.png
[output_9_1]: /assets/output_9_1.png
[output_17_0]: /assets/output_17_0.png
[output_24_0]: /assets/output_24_0.png
[output_26_0]: /assets/output_26_0.png
[output_30_1]: /assets/output_30_1.png
[output_63_0]: /assets/output_63_0.png
[output_75_0]: /assets/output_75_0.png
[output_75_1]: /assets/output_75_1.png
[output_75_2]: /assets/output_75_2.png
[output_75_3]: /assets/output_75_3.png
[output_75_4]: /assets/output_75_4.png
[output_75_5]: /assets/output_75_5.png
[graph]: /assets/graph.png



## Project: Build a Traffic Sign Recognition Classifier

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'datasets/train.p'
testing_file = 'datasets/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
```

### Step 0.1: construct the class lookup table


```python
import pandas as pd
sign_lookup = pd.read_csv('signnames.csv')
```


```python
sign_lookup
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClassId</th>
      <th>SignName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Speed limit (20km/h)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Speed limit (30km/h)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Speed limit (50km/h)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Speed limit (60km/h)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Speed limit (70km/h)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Speed limit (80km/h)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>End of speed limit (80km/h)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Speed limit (100km/h)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Speed limit (120km/h)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>No passing</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>No passing for vehicles over 3.5 metric tons</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Right-of-way at the next intersection</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Priority road</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Yield</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Stop</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>No vehicles</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Vehicles over 3.5 metric tons prohibited</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>No entry</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>General caution</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Dangerous curve to the left</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>Dangerous curve to the right</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>Double curve</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>Bumpy road</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>Slippery road</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>Road narrows on the right</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>Road work</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>Traffic signals</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>Pedestrians</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>Children crossing</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>Bicycles crossing</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>Beware of ice/snow</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>Wild animals crossing</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>End of all speed and passing limits</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>Turn right ahead</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>Turn left ahead</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>Ahead only</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>Go straight or right</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>Go straight or left</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>Keep right</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>Keep left</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40</td>
      <td>Roundabout mandatory</td>
    </tr>
    <tr>
      <th>41</th>
      <td>41</td>
      <td>End of no passing</td>
    </tr>
    <tr>
      <th>42</th>
      <td>42</td>
      <td>End of no passing by vehicles over 3.5 metric ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
sign_lookup.loc[0]['SignName']
```




    'Speed limit (20km/h)'



---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below.


```python
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

fig, axs = plt.subplots(2,5, figsize=(10, 3))
axs = axs.ravel()
plt.rc('font', size=5)
for i in range(10):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title("{}: {}".format(y_train[index], sign_lookup.loc[y_train[index]]['SignName']))
```


![alt text][output_8_0]



```python
from collections import Counter
import matplotlib.patches as mpatches
import seaborn as sns

c = Counter(y_train)
c2 = Counter(y_test)

plt.bar(list(c.keys()),list(c.values()))
plt.bar(list(c2.keys()),list(c2.values()),color='green')
blue_patch = mpatches.Patch(color='blue', label='Train data')
green_patch = mpatches.Patch(color='green', label='Test data')
plt.xticks(list(c.keys()), sign_lookup['SignName'],rotation='90')
plt.legend(handles=[blue_patch,green_patch])
```




    <matplotlib.legend.Legend at 0x7f3018749b00>



![alt text][output_9_1]


The dataset is imbalanced in term of classes though, it might be a good idea to generate data that is lower than 900


```python
### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = image.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(sign_lookup)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43



#### Preprocessing idea
- If the images are not very bright, pre-processing techniques like histogram equalization can help
 - Compare Adaptive & normal histrogram EQ
- Some other techniques are mean subtraction and standard deviation scaling of the input data
 - uses z-score
- More preprocessing steps could have been included
- This resource might provide some more intuition on the subjec


```python
from scipy import stats
from skimage import exposure
from tqdm import tqdm

# This function taken from
# http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.array(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]), dtype=np.uint8)

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # from http://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
    
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape), cdf
```


```python
X_train_gray = rgb2gray(X_train)
X_test_gray = rgb2gray(X_test)

X_train_eq = np.zeros(X_train_gray.shape)
X_test_eq = np.zeros(X_test_gray.shape)
print(X_train_gray.shape)
print(X_train_eq.shape)

for i in tqdm(range(X_train_gray.shape[0])):
    X_train_eq[i] = exposure.equalize_adapthist(X_train_gray[i])
    
for i in tqdm(range(X_test_gray.shape[0])):
    X_test_eq[i] = exposure.equalize_adapthist(X_test_gray[i])
    
X_train_eq_simple = image_histogram_equalization(X_train_gray)[0]
X_test_eq_simple = image_histogram_equalization(X_test_gray)[0]
```

      0%|          | 21/39209 [00:00<03:07, 208.88it/s]

    (39209, 32, 32)
    (39209, 32, 32)


    100%|██████████| 39209/39209 [03:10<00:00, 205.78it/s]
    100%|██████████| 12630/12630 [01:01<00:00, 205.83it/s]


### Visualization between Adaptive Histrogram Equalization and normal Histrogram equalization


```python
fig, axs = plt.subplots(3,5, figsize=(10, 3))
axs = axs.ravel()
plt.rc('font', size=5)
for i in range(5):
    index = random.randint(0, len(X_train_eq))
    image = X_train_eq[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title("{}: {}".format(y_train[index], sign_lookup.loc[y_train[index]]['SignName']))
    
    image = X_train_gray[index].squeeze()
    axs[i+5].axis('off')
    axs[i+5].imshow(image, cmap='gray')
    
    image = X_train_eq_simple[index].squeeze()
    axs[i+10].axis('off')
    axs[i+10].imshow(image, cmap='gray')
```


![alt text][output_17_0]

### It seems normal histrogram equalization is better, let's take this one


```python
X_train_normalized = stats.zscore(X_train_eq_simple)
X_test_normalized = stats.zscore(X_test_eq_simple)
```

### now, let's take care of those imbalanced classes, let's generate!


```python
class_counter = Counter(y_train)
```


```python
need_gen = [i for i in range(43) if np.array(list(class_counter.values()))[i] < 900]
```

## Warp testing


```python
from skimage.transform import rotate, warp

def random_shift(xy):
    xy[:, 0] += np.random.uniform(-5,5)
    xy[:, 1] += np.random.uniform(-5,5)
    return xy

fig, axs = plt.subplots(2,5, figsize=(10, 3))
axs = axs.ravel()
plt.rc('font', size=5)
for i in range(5):
    index = random.randint(0, len(X_train))
    image = X_train_normalized[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title("{}: {}".format(y_train[index], sign_lookup.loc[y_train[index]]['SignName']))
    
    image = warp(X_train_normalized[index], random_shift)
    axs[i+5].axis('off')
    axs[i+5].imshow(image, cmap='gray')
```

![alt text][output_24_0]


## Rotation testing 


```python
from skimage.transform import rotate, warp_coords
fig, axs = plt.subplots(2,5, figsize=(10, 3))
axs = axs.ravel()
plt.rc('font', size=5)
for i in range(5):
    index = random.randint(0, len(X_train))
    image = X_train_normalized[index].squeeze()
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title("{}: {}".format(y_train[index], sign_lookup.loc[y_train[index]]['SignName']))
    
    image = rotate(X_train_normalized[index], angle=np.random.uniform(-15,15))
    axs[i+5].axis('off')
    axs[i+5].imshow(image, cmap='gray')
```

![alt text][output_26_0]



```python
print("Before Generate: X={} Y={}".format(X_train_normalized.shape, y_train.shape))
```

    Before Generate: X=(39209, 32, 32) Y=(39209,)



```python
# This chunk of code on how to append result to the original one I got the idea from Jeremy Shannon's code
# https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb
class_sample_thres = 900
for _class in tqdm(need_gen):
    populations = np.where(y_train == _class)[0]
    nb_samples = len(populations)
    if nb_samples < class_sample_thres:
        for i in range(abs(nb_samples - class_sample_thres)):
            # get one from the dataset
            gen_img = X_train_normalized[populations[i % nb_samples]]
            # augment it
            gen_img = warp(rotate(gen_img, angle=np.random.uniform(-15,15)), random_shift)
            # append it to X,Y
            X_train_normalized = np.concatenate((X_train_normalized, [gen_img]), axis=0)
            y_train = np.concatenate((y_train, [_class]), axis=0)
```

    100%|██████████| 26/26 [09:53<00:00, 30.60s/it]



```python
print("After Generate: X={} Y={}".format(X_train_normalized.shape, y_train.shape))
```

    After Generate: X=(52110, 32, 32) Y=(52110,)



```python
c = Counter(y_train)

plt.bar(list(c.keys()),list(c.values()), color='blue')
blue_patch = mpatches.Patch(color='blue', label='Train data')
plt.xticks(list(c.keys()), sign_lookup['SignName'],rotation='90')
plt.legend(handles=[blue_patch])
```




    <matplotlib.legend.Legend at 0x7f30186b6c50>




![alt text][output_30_1]


### Every dataset I'm shuffling


```python
y_train_temp = y_train

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.2)

for train_index, test_index in sss.split(X_train_normalized, y_train_temp):
    X_train, X_validation = X_train_normalized[train_index], X_train_normalized[test_index]
    y_train, y_validation = y_train_temp[train_index], y_train_temp[test_index]
```


```python
X_train.shape
```




    (41688, 32, 32)




```python
X_validation.shape
```




    (10422, 32, 32)




```python
X_test = X_test_normalized

X_train = np.reshape(X_train,[41688,32,32,1])
X_validation = np.reshape(X_validation,[10422,32,32,1])
X_test = np.reshape(X_test_normalized,[12630,32,32,1])

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)
```

    (41688, 32, 32, 1)
    (10422, 32, 32, 1)
    (12630, 32, 32, 1)


----

## Step 2: Design and Test a Model Architecture

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

## TF Things


```python
import tensorflow as tf

EPOCHS = 60
BATCH_SIZE = 128
```


```python
from tensorflow.contrib.layers import flatten

def LeNetInception(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    with tf.name_scope('Convolution'):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 6), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    with tf.name_scope('Relu'):
        conv1 = tf.nn.relu(conv1)

    with tf.name_scope('Convolution'):
        conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    with tf.name_scope('Relu'):
        conv2 = tf.nn.relu(conv2)
    
    with tf.name_scope('Convolution'):
        conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 20), mean = mu, stddev = sigma))
        conv3_b = tf.Variable(tf.zeros(20))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    with tf.name_scope('Relu'):
        conv3 = tf.nn.relu(conv3)
    
    with tf.name_scope('Convolution'):
        conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 20, 28), mean = mu, stddev = sigma))
        conv4_b = tf.Variable(tf.zeros(28))
        conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b

    with tf.name_scope('Relu'):
        conv4 = tf.nn.relu(conv4)
    
    with tf.name_scope('Maxpool'):
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    with tf.name_scope('Flatten'):
        fc0   = flatten(conv4)
    
    with tf.name_scope('FC1'):
        fc1_W = tf.Variable(tf.truncated_normal(shape=(2800, 120), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    with tf.name_scope('Relu'):
        fc1    = tf.nn.relu(fc1)
    
    with tf.name_scope('Dropout'):
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('FC2'):
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(84))
        fc2    = tf.matmul(fc1_drop, fc2_W) + fc2_b
    
    with tf.name_scope('Relu'):
        fc2    = tf.nn.relu(fc2)
    
    with tf.name_scope('Dropout'):
        fc2_drop = tf.nn.dropout(fc2, keep_prob)
    
    with tf.name_scope('FC3'):
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2_drop, fc3_W) + fc3_b
    
    return logits
```


```python
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="x-input") 
    y = tf.placeholder(tf.int32, (None), name="y-input") 
    one_hot_y = tf.one_hot(y, 43)
    
with tf.name_scope('Dropoutprob'):
    keep_prob = tf.placeholder(tf.float32)
```


```python
rate = 0.001

logits = LeNetInception(x)
with tf.name_scope('Softmax'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
with tf.name_scope('Loss'):
    loss_operation = tf.reduce_mean(cross_entropy)
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
with tf.name_scope('Minimize'):
    training_operation = optimizer.minimize(loss_operation)
```


```python
with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
with tf.name_scope('accuracy_operation'):
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```


```python
from sklearn.utils import shuffle
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
logs_path = "/tmp/traffic_sign/1"
                  
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.785
    
    EPOCH 2 ...
    Validation Accuracy = 0.860
    
    EPOCH 3 ...
    Validation Accuracy = 0.902
    
    EPOCH 4 ...
    Validation Accuracy = 0.917
    
    EPOCH 5 ...
    Validation Accuracy = 0.934
    
    EPOCH 6 ...
    Validation Accuracy = 0.942
    
    EPOCH 7 ...
    Validation Accuracy = 0.953
    
    EPOCH 8 ...
    Validation Accuracy = 0.958
    
    EPOCH 9 ...
    Validation Accuracy = 0.964
    
    EPOCH 10 ...
    Validation Accuracy = 0.964
    
    EPOCH 11 ...
    Validation Accuracy = 0.967
    
    EPOCH 12 ...
    Validation Accuracy = 0.970
    
    EPOCH 13 ...
    Validation Accuracy = 0.970
    
    EPOCH 14 ...
    Validation Accuracy = 0.972
    
    EPOCH 15 ...
    Validation Accuracy = 0.972
    
    EPOCH 16 ...
    Validation Accuracy = 0.974
    
    EPOCH 17 ...
    Validation Accuracy = 0.974
    
    EPOCH 18 ...
    Validation Accuracy = 0.978
    
    EPOCH 19 ...
    Validation Accuracy = 0.980
    
    EPOCH 20 ...
    Validation Accuracy = 0.978
    
    EPOCH 21 ...
    Validation Accuracy = 0.978
    
    EPOCH 22 ...
    Validation Accuracy = 0.976
    
    EPOCH 23 ...
    Validation Accuracy = 0.975
    
    EPOCH 24 ...
    Validation Accuracy = 0.979
    
    EPOCH 25 ...
    Validation Accuracy = 0.981
    
    EPOCH 26 ...
    Validation Accuracy = 0.981
    
    EPOCH 27 ...
    Validation Accuracy = 0.982
    
    EPOCH 28 ...
    Validation Accuracy = 0.983
    
    EPOCH 29 ...
    Validation Accuracy = 0.982
    
    EPOCH 30 ...
    Validation Accuracy = 0.981
    
    EPOCH 31 ...
    Validation Accuracy = 0.983
    
    EPOCH 32 ...
    Validation Accuracy = 0.984
    
    EPOCH 33 ...
    Validation Accuracy = 0.983
    
    EPOCH 34 ...
    Validation Accuracy = 0.985
    
    EPOCH 35 ...
    Validation Accuracy = 0.985
    
    EPOCH 36 ...
    Validation Accuracy = 0.983
    
    EPOCH 37 ...
    Validation Accuracy = 0.983
    
    EPOCH 38 ...
    Validation Accuracy = 0.984
    
    EPOCH 39 ...
    Validation Accuracy = 0.982
    
    EPOCH 40 ...
    Validation Accuracy = 0.985
    
    EPOCH 41 ...
    Validation Accuracy = 0.985
    
    EPOCH 42 ...
    Validation Accuracy = 0.985
    
    EPOCH 43 ...
    Validation Accuracy = 0.984
    
    EPOCH 44 ...
    Validation Accuracy = 0.983
    
    EPOCH 45 ...
    Validation Accuracy = 0.985
    
    EPOCH 46 ...
    Validation Accuracy = 0.987
    
    EPOCH 47 ...
    Validation Accuracy = 0.986
    
    EPOCH 48 ...
    Validation Accuracy = 0.982
    
    EPOCH 49 ...
    Validation Accuracy = 0.985
    
    EPOCH 50 ...
    Validation Accuracy = 0.986
    
    EPOCH 51 ...
    Validation Accuracy = 0.984
    
    EPOCH 52 ...
    Validation Accuracy = 0.981
    
    EPOCH 53 ...
    Validation Accuracy = 0.985
    
    EPOCH 54 ...
    Validation Accuracy = 0.982
    
    EPOCH 55 ...
    Validation Accuracy = 0.986
    
    EPOCH 56 ...
    Validation Accuracy = 0.985
    
    EPOCH 57 ...
    Validation Accuracy = 0.987
    
    EPOCH 58 ...
    Validation Accuracy = 0.985
    
    EPOCH 59 ...
    Validation Accuracy = 0.986
    
    EPOCH 60 ...
    Validation Accuracy = 0.984
    
    Model saved


Log

- 20epoch GrayScale + Zscore, Lenet architecture validation_acc = 0.979, test = 0.882
- 60epoch GrayScale + Zscore, 3x3 3x3 5x5 5x5 maxpool architecture validation_acc = 0.996, test = 0.968
- 60epoch GrayScale + histeq + Zscore + data generation(rotate,warp) validation_acc = 0.986, test = 0.971


```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, "./lenet")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
```

    Test Set Accuracy = 0.970


**Data Preprocessing**
 - rgb to grayscale, my perception is that the Traffic sign should reproduce the same meaning even it is in grayscale, and thus should cover the same procedure because the convnet is trying to mimic human's eyes
 - Histrogram equalization on the images to make sure that the images are not too bright or too dark 
 - i Normalize the data using zscore to prevent gradients explode

**Train/Validate/Test Splitting**
 - I use StratifiedShuffleSplit for splitting dataset for train/validate as mentioned from the suggestions. I use test_size=0.2
 
**Data Generation**
  - How Did I generate data
    - I generate the data from original dataset by
      - rotate the images by ±15°
      - translate(move) the images up down left right by ±5 pixels
  - Why Did I generate data
    - The numbers of data classes are imbalanced, so I try to generate them which prevents the problems when training the network
  - The differences from original dataset
    - we got almost balanced dataset for most classes, it's not that balanced, but at least it's much better then before

**Final Architecture**

Here is the image from my TensorBoard
![Architecture from TensorBoard][graph]

the font rendering is not stable and it lacks of some information, please take a look on the list below

My architecture is as the following

1. 6@3x3 convolution
1. 16@3x3 convolution
1. 20@5x5 convolution
1. 28@5x5 convolution
1. 2x2 maxpooling
1. (2800, 120) Fully Connected
1. Dropout keep_prob = 0.7
1. (120, 84) Fully Connected
1. Dropout keep_prob = 0.7
1. (84, 43) Fully Connected

Note that I did not use the maxpooling between convolutional layers, becuase I feel that making convolution to get all the features might be more useful & be able to extract more features even it is more computational expensive

The main reason I use dropout is because [this](Dropout: A Simple Way to Prevent Neural Networks from Overfitting) paper states that using dropout a network can be able to prevent complex co-adaptation, and I gives higher test accuracy :)

### Model Training 4

In summary,

1. EPOCHS = 60
1. BATCHSIZE = 128
1. learning_rate = 0.001
1. keep_prob =0.7
1. AdamOptimizer


Mostly are the same, all i added is just to train for more epochs, at first i am scared that it might lead to overfitting. but I got dropout, so at least it could prevent it. I train module vary the keep_prob from 0.5 0.6 0.7 and I found that 0.7 gives me the best value

### Approches to Solution

Mostly Trial and error

 - I use AdamOptimizer because it is like Stochastic Gradient Descent with smaller memory requirement and provides adaptive learning rate, According to ADAM: A method for stochastic optimization. 
 - I chose the type of layer under the perception that trying to vary the size of convolutional kernel from small to big is the right thing to do because the small kernel it would track the abstraction which might be features of a particular class and the bigger kernel might extract the high level feature
 - there are 5 hyperparameters I need to tune which are EPOCHS, BATCHSIZE, learning_rate and dropout keep_prob
  - for the first 4 I try to do all combinations of them and see which one gives the best result
  - then, I try to vary the value of keep_prob based on how it does well on the test set
 - I tested the dataset with vanilla lenet, but it does not generalize well on the test&my dataset, so I believe the network lacks of some data. That's why I try to generate more dataset and preprocess them according to my sense that the images can be rotate and transform according to the raw data
 - I fine-tuned the model upon how well the model works on validation set, does it converges too fast? or too slow, once the model can proceed train well, I finalize the result based on benchmark it on the accuracy of test set. I also check that the result of training and testing dataset should not be too far, or it might cause overfitting


---

## Test a Model on New Images



```python
from glob import glob 
import matplotlib.image as mpimg

from skimage import color
from skimage import io

images = glob('my-images/*.png')
fig, axs = plt.subplots(2,5, figsize=(10, 3))
axs = axs.ravel()
new_imgs = []
for idx,img_path in enumerate(images):
    # I use rgb2gray from skimage because the original method doesnot work with this additional dataset
    new_img_rgb = io.imread(img_path)
    new_img_gray= color.rgb2gray(new_img_rgb);
    new_img_sq_simple = image_histogram_equalization(new_img_gray)[0]
    new_img = stats.zscore(new_img_sq_simple)
    new_imgs.append(new_img)
    axs[idx].axis('off')
    axs[idx].imshow(new_img_rgb, cmap='gray')
```


![alt text][output_63_0]



```python
y_new_imgs = [18,29,35,14,31,31]
```


```python
new_imgs = np.asarray(new_imgs)
new_imgs_nonreshape = new_imgs
new_imgs = np.reshape(new_imgs,[6,32,32,1])
new_imgs.shape
```




    (6, 32, 32, 1)



### Which images make classification difficults

There are general cases where the images are hard to classified which are
 - The Brightness, the image may be too dark to perceive information from them
 - The Contrast, the image might lack of contrast to extract detail
 - The Angle, for some sign, the rotation causes confusion to the network and human
 
the images are on the above cells, I find difficulties for most images i chose
 - 1st image, nothing is wrong, the network should be able to classify this
 - 2nd image, the bicycle crossing one, I think it is hard to classify since the data is downsampling poorly by paint which can't even connect the wheel, and there is a person on the sign which is not exact the same as we have in dataset
 - 3rd image, the go ahead, the network should be able to classify this
 - 4th image, the stop, if my network learn to detect the "STOP" font, it should be able to classify this easily, however since the network might pick up the 6 corner shape(pentagon?) and the lower part of the image is cut, it might create problem with the classifier
 - 5th & 6th image, the wildlife ones, I'm not sure if we have enough data to train with different species with the network, both two surely break the network :D

### Accuracy on real-world


```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    saver3 = tf.train.import_meta_graph('./lenet.meta')
    saver3.restore(sess, "./lenet")
    new_accuracy = evaluate(new_imgs, y_new_imgs)
    print("Test Set Accuracy = {:.3f}".format(new_accuracy))
```

    Test Set Accuracy = 0.167



My model is too weak to generalize and localize the street sign I found on Google :D. The Accuracy on my acquired images is quite low at only **16.7%** while it was **97.0%** on the testing set. I believe the reason can be found in the answer to the question 6 or the model might be overfitting in the sense of it lacks variety of data (camels,cows)

### Certainties of the predictions

For the centainty of predictions, the model is certain and correctly predicted only the one obvious sign which is Go Ahead only (3rd image). However, when it comes to other signs, the network is too far to be correct for these images. I believe the reason is explained on the answer to question 6. There is a case which is the 1st image(General Caution) that its class appears on the top_k predictions, but the softmax probability is too little to be correct :D, and I hope it was obvious and could be predicted correctly. Maybe, it was because I didn't downsampling it nicely.


```python
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    saver4 = tf.train.import_meta_graph('./lenet.meta')
    saver4.restore(sess, "./lenet")
    new_softmax_logits = sess.run(softmax_logits, feed_dict={x: new_imgs, keep_prob: 1.0})
    top_predictions = sess.run(top_k, feed_dict={x: new_imgs, keep_prob: 1.0})
```


```python
images = glob('my-images/*.png')
    
# I took code from the suggestions
for i,(labels,probs,img_path) in enumerate(zip(top_predictions.indices, top_predictions.values, images)):
    fig = plt.figure(figsize=(15,2))
    plt.bar(labels,probs)
    plt.title(sign_lookup.loc[y_new_imgs[i]]['SignName'])
    img = mpimg.imread(img_path)
    height = img.shape[0]
    plt.xticks(np.arange(0.5, 43.5, 1.0), sign_lookup['SignName'].values, ha='right', rotation=45)
    plt.yticks(np.arange(0.0, 1.0, 0.1), np.arange(0.0,1.0,0.1))
    
    ax = plt.axes([.75, 0.25, 0.5, 0.5], frameon=True)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    
plt.show()
    
```


![alt text][output_75_0]



![alt text][output_75_1]



![alt text][output_75_2]



![alt text][output_75_3]



![alt text][output_75_4]



![alt textng][output_75_5]