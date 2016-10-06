
# coding: utf-8

# In[26]:

import numpy as np
import os, re
from scipy import ndimage, misc

from infogan.misc.datasets import Dataset


# In[27]:

def load_all_images_from_dir(directory_path, num_images_to_grab = None, crop_size = (32,32)):
    '''
    num_images_to_grab -- if None, load all images in directory_path, else only load up to num_images_to_grab.
    crop_size -- crops all images to crop_size.
    '''
    images = []
    for root, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if num_images_to_grab is not None and len(images) >= num_images_to_grab:
                break
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                filepath = os.path.join(root, filename)
                image = ndimage.imread(filepath, mode="RGB")
                image_resized = misc.imresize(image, crop_size)
                images.append(image_resized)
    return images

def ParitionData(images):
    num_total_inputs = len(images)

    train_images = images[0:num_total_inputs*6/10]
    cv_images    = images[num_total_inputs*6/10:num_total_inputs*8/10]
    test_images  = images[num_total_inputs*8/10:]
    
    return train_images, cv_images, test_images


# In[35]:

class CelebADataset(object):
    def __init__(self, num_images_to_grab = None):
        data_directory = "celebA/img_align_celeba/"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        self.raw_images = load_all_images_from_dir(data_directory, num_images_to_grab, (32,32))
        train_images, cv_images, test_images = ParitionData([x.flatten() for x in self.raw_images])

        self.train = Dataset(np.asarray(train_images))
        self.validation = Dataset(np.asarray(cv_images))
        self.test = Dataset(np.asarray(test_images))

        ## make sure that each type of digits have exactly 10 samples
        #sup_images = []
        #sup_labels = []
        #rnd_state = np.random.get_state()
        #np.random.seed(0)
        #for cat in range(10):
        #    ids = np.where(self.train.labels == cat)[0]
        #    np.random.shuffle(ids)
        #    sup_images.extend(self.train.images[ids[:10]])
        #    sup_labels.extend(self.train.labels[ids[:10]])
        #np.random.set_state(rnd_state)
        #self.supervised_train = Dataset(
        #    np.asarray(sup_images),
        #    np.asarray(sup_labels),
        #)
        
        self.image_dim = 32*32*3
        self.image_shape = (32,32,3)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


# In[36]:

partial_dataset = CelebADataset(10)


# In[37]:

partial_dataset.train.images.shape


# In[ ]:

32*32*3


# In[ ]:




# In[ ]:



