import random
import cv2
import numpy as np
from Augmentor.Operations import Operation
from PIL import Image


class Patcher(Operation):
    """
    This class is used to Color images .
    """

    def __init__(self, probability):
        """
        As the aspect ratio is always kept constant, only a
        :attr:`scale_factor` is required for scaling the image.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param scale_factor: The factor by which to scale, where 1.5 would
         result in an image scaled up by 150%.
        :type probability: Float
        :type scale_factor: Float
        """
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Scale the passed :attr:`images` by the factor specified during
        instantiation, returning the scaled image.
        :param images: The image to scale.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            source_im = image if augmented_images == [] else random.choice(augmented_images)
            start = [source_im.size[0] // random.randint(1, 5), source_im.size[0] // random.randint(1, 5)]
            e1 = source_im.size[0] // random.randint(1, 5)
            e2 = source_im.size[0] // random.randint(1, 5)
            end = [min(start[0] + e1, source_im.size[0]), min(start[1] + e1, source_im.size[0])]
            region = source_im.crop([*start, *end])

            start2 = [image.size[0] // random.randint(1, 5), image.size[0] // random.randint(1, 5)]
            end2 = [min(start2[0] + e1, image.size[0]), min(start2[1] + e1, image.size[0])]
            image.paste(region, [*start2, *end2])
            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
