from scipy.misc import imread
from random import shuffle
import time
import subprocess as sp

import tensorflow as tf
from glob import glob
from utils import get_image, colorize
# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 128
FFMPEG_BIN = 'ffmpeg'
frame_dim = (720,1280,3)  # height x width x num_colors(depth)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(argv):
    command = [ FFMPEG_BIN,
                '-i', 'andrew_ng_speaking_cropped.mp4',
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-']
    video_pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

    # read 1 frame bytes
    raw_image = video_pipe.stdout.read(frame_dim[0]*frame_dim[1]*frame_dim[2])
    print "Raw image bytes =", len(raw_image)
    # transform the byte read into a numpy array
    image =  np.fromstring(raw_image, dtype='uint8')
    image = image.reshape(frame_dim)
    # throw away the data in the pipe's buffer.
    pipe.stdout.flush()

    outfile = 'ng_video_frames/frame' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for i, f in enumerate(files):
        print i
        image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        #print image.min(), image.max()
        # from pylearn2.utils.image import save
        # save('foo.png', (image + 1.) / 2.)
        image_raw = image.tostring()
        class_str = f.split('/')[-2]
        label = str_to_int[class_str]
        if i % 1 == 0:
            print i, '\t',label
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    tf.app.run()

