from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets import MnistDataset
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime
import scipy.misc

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"
    root_generated_images_dir = "gen/mnist"

    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 50

    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)
    gen_dir = os.path.join(root_generated_images_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(gen_dir)

    dataset = MnistDataset()

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    sess = tf.Session()
    algo.train(sess)

    # Use the trained model to generate one batch of images!
    generated_images = sess.run(algo.fake_x)

    print("Saving generated images to " + gen_dir)

    for i,img in enumerate(generated_images):
        img_file = os.path.join(gen_dir, 'generated_%05d.png' % (i))
        scipy.misc.toimage(img.reshape(dataset.image_shape[:2]), cmin=0.0, cmax=1.0).save(img_file)

