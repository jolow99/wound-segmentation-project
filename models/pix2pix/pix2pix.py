
import tensorflow as tf
import sys
sys.path.append('../..')
from IPython import display
import time
import datetime
import os
from scripts.dataloader import TFDataGen, create_tf_datasets
from matplotlib import pyplot as plt


class Pix2pix():
    def __init__(self, lamb):
        self.lamda = lamb
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.Generator()
        self.discriminator =self.Discriminator()
        self.summary_writer = tf.summary.create_file_writer("logs/" + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

    def downsample(self,filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 112, 112, 64)
            self.downsample(128, 4),  # (batch_size, 56, 56, 128)
            self.downsample(256, 4),  # (batch_size, 28, 28, 256)
            self.downsample(512, 4),  # (batch_size, 14, 14, 512)
            self.downsample(512, 4),  # (batch_size, 7, 7, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 14, 14, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 28, 28, 1024)
            self.upsample(256, 4),  # (batch_size, 56, 56, 512)
            self.upsample(128, 4),  # (batch_size, 112, 112, 256)
            self.upsample(64, 4),  # (batch_size, 224, 224, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.lamb * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    
    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[224, 224, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[224, 224, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 224, 224, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 112, 112, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 56, 56, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 28, 28, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 30, 30, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 27, 27, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 29, 29, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 26, 26, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    
    def discriminator_loss(self,disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
    def train_step(self,input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

    def generate_images(self,model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def fit(self,train_ds, test_ds, steps):
        # example_input, example_target = next(iter(test_ds))
        start = time.time()
        step = 0
        while step < steps:
            for input_image, target in test_ds:
                print(input_image.shape)
                if step >= steps:
                    break

                # Your training code here
                # Remove train_ds.repeat().take(steps).enumerate()
                # and replace with the code above

                self.train_step(input_image, target, step)

                step += 1
                if step % 1000 == 0:
                    # Code to evaluate on test_loader or any logging you want to do
                    display.clear_output(wait=True)
                    if step != 0:
                        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
                    start = time.time()
                    self.generate_images(self.generator, input_image, target)
                    print(f"Step: {step//1000}k")
                    pass

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)


dataloader = TFDataGen('/Users/jefflai/Desktop/ADL/wound-segmentation-project/data/wound_dataset/azh_wound_care_center_dataset_patches/',0.8, 224, 224, 'rgb')
train_ds, valid_ds, test_ds = create_tf_datasets(dataloader, 1)
model = Pix2pix(100)
model.fit(train_ds, test_ds, 4000)