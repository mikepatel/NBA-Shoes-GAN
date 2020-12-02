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

    # optimizers

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
    d = build_discriminator()
    d.summary()

    g = build_generator()
    g.summary()

    quit()

    # ----- TRAIN ----- #
    train(
        discriminator=d,
        generator=g,
        dataset=dataset
    )
