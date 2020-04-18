"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For generating a gif of training epochs
"""
################################################################################
# Imports
import os
import glob
import imageio


################################################################################
# Main
if __name__ == "__main__":
    # ----- VISUALIZATION ----- #
    results_dir = input("Enter images directory: ")
    gif_filename = os.path.join(os.getcwd(), "training.gif")

    # get all images
    image_files_pattern = results_dir + "\\*.png"
    filenames = glob.glob(image_files_pattern)

    # write all images to gif
    with imageio.get_writer(gif_filename, mode="I", fps=1.0) as writer:  # 'I' for multiple images
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
