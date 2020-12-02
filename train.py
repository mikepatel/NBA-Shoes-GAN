"""


"""
################################################################################
# Imports
from parameters import *
from model import build_discriminator, build_generator


################################################################################
# generate and save images
def generate_and_save_images(model, epoch, z_input, save_dir):
    predictions = model(z_input, training=False)
    predictions = predictions[:NUM_GEN_IMAGES]  # generate 16 images

    # rescale from [-1, 1] to [0, 1]
    #predictions = (predictions + 1) / 2

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

    # generator noise seed (in order to visualize training)

    # create output directory for results

    # training loop
    print()
        # generate images while training

        # train over batch

    # generate one more image for the last epoch


################################################################################
# Main
if __name__ == "__main__":
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
    d.summary()

    quit()

    # ----- TRAIN ----- #
    train(
        discriminator=d,
        generator=g,
        dataset=dataset
    )
