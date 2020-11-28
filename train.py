"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
from parameters import *
from model import build_discriminator, build_discriminator_vgg16, build_generator


################################################################################
# generate and save images
def generate_and_save_images(model, epoch, z_input, save_dir):
    predictions = model(z_input, training=False)
    predictions = predictions[:16]  # generate 16 images

    # rescale from [-1, 1] to [0, 1]
    predictions = (predictions + 1) / 2

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:05d}')
    plt.savefig(fig_name)
    plt.close()


# training loop
def train(generator, discriminator, dataset):

    # loss functions
    def discriminator_loss_fn(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss_fn(fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        return loss

    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(real_batch):
        noise = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_images = generator(noise, training=True)

            real_output = discriminator(real_batch, training=True)
            fake_output = discriminator(fake_images, training=True)

            generator_loss = generator_loss_fn(fake_output=fake_output)
            discriminator_loss = discriminator_loss_fn(real_output=real_output, fake_output=fake_output)

        generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # generator noise seed (in order to visualize training)
    noise_seed = tf.random.normal(shape=(NUM_GEN_IMAGES, NOISE_DIM))

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # training loop
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch}')

        # train over batch
        for batch in dataset:
            train_step(batch)

        # generate images while training
        generate_and_save_images(
            model=generator,
            epoch=epoch,
            z_input=noise_seed,
            save_dir=output_dir
        )


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    #train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)  # og shape is (50k, 32, 32, 3)
    train_images = train_images.astype(np.float32)

    # rescale from [0, 255] to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    print(train_images.shape)

    # dataset
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.shuffle(buffer_size=train_images.shape[0])
    dataset = dataset.batch(batch_size=BATCH_SIZE)

    # ----- MODEL ----- #
    g = build_generator()
    g.summary()

    """
    noise = tf.random.normal(shape=(1, NOISE_DIM))
    z = g(noise, training=False)
    z = tf.squeeze(z, axis=0)
    z = tf.keras.preprocessing.image.array_to_img(z)
    z.show()
    """
    d = build_discriminator()
    d.summary()

    # ----- TRAIN ----- #
    train(
        generator=g,
        discriminator=d,
        dataset=dataset
    )
