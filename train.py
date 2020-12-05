"""


"""
################################################################################
# Imports
from parameters import *
from model import build_discriminator, build_generator, build_discriminator_vgg16


################################################################################
# generate and save images
def generate_and_save_images(model, epoch, z_input, save_dir):
    predictions = model(z_input, training=False)
    predictions = predictions[:NUM_GEN_IMAGES]  # generate 16 images

    # rescale from [-1, 1] to [0, 1]
    predictions = (predictions + 1.0) / 2.0

    for i in range(NUM_GEN_IMAGES):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:05d}')
    plt.savefig(fig_name)
    plt.close()


# train
def train(discriminator, generator, dataset):
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
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    # train on batch
    #@tf.function
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

        # generate images while training
        generate_and_save_images(
            model=generator,
            epoch=epoch,
            z_input=noise_seed,
            save_dir=output_dir
        )

        # train over batch
        for i in range(len(dataset)):
            batch = dataset[i]
            train_step(batch)

    # generate one more image for the last epoch
    generate_and_save_images(
        model=generator,
        epoch=NUM_EPOCHS,
        z_input=noise_seed,
        save_dir=output_dir
    )

    # save trained generator model
    save_model_filepath = SAVED_MODEL_DIR
    generator.save(save_model_filepath)


################################################################################
# Main
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rescale=1./255
    )

    dataset = image_generator.flow_from_directory(
        directory=DATA_DIR,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        classes=None,
        class_mode=None,
        shuffle=True
    )

    """
    x = next(dataset)
    x = x[:16]
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(x[i])
        plt.axis("off")

    plt.savefig("x")
    plt.close()
    quit()
    """

    # ----- MODEL ----- #
    g = build_generator()
    g.summary()

    d = build_discriminator()
    #d = build_discriminator_vgg16()
    d.summary()

    """
    noise = tf.random.normal(shape=(1, NOISE_DIM))
    x = g(inputs=tf.random.normal(shape=(1, NOISE_DIM)), training=False)
    x = x[0]
    print(x)
    x = (x + 1) / 2
    plt.imshow(x)
    plt.savefig("x")
    quit()
    """

    # ----- TRAIN ----- #
    train(
        discriminator=d,
        generator=g,
        dataset=dataset
    )
