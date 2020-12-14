from Augmentor.Operations import Operation
import random
from PIL import Image
from math import floor, ceil

class RotateZoom(Operation):
    """
    This class is used to perform rotations on images in multiples of 90
    degrees. Arbitrary rotations are handled by the :class:`RotateRange`
    class.
    """

    def __init__(self, probability, rotation):
        """
        As well as the required :attr:`probability` parameter, the
        :attr:`rotation` parameter controls the rotation to perform,
        which must be one of ``90``, ``180``, ``270`` or ``-1`` (see below).

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param rotation: Controls the rotation to perform. Must be one of
         ``90``, ``180``, ``270`` or ``-1``.

         - ``90`` rotate the image by 90 degrees.
         - ``180`` rotate the image by 180 degrees.
         - ``270`` rotate the image by 270 degrees.
         - ``-1`` rotate the image randomly by either 90, 180, or 270 degrees.

        .. seealso:: For arbitrary rotations, see the :class:`RotateRange` class.

        """
        Operation.__init__(self, probability)
        self.rotation = rotation

    def __str__(self):
        return "Rotate " + str(self.rotation)

    def perform_operation(self, images):
        """
        Rotate an image by either 90, 180, or 270 degrees, or randomly from
        any of these.

        :param images: The image(s) to rotate.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        random_factor = random.randint(1, 3)
        z_factor = random.randint(1, 5)
        def do(image):
            w, h = image.size
            if self.rotation == -1:
                image = image.rotate(90 * random_factor, expand=True)
            else:
                image =  image.rotate(self.rotation, expand=True)


            image_zoomed = image.resize((int(round(image.size[0] * z_factor)),
                                         int(round(image.size[1] * z_factor))),
                                        resample=Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            image = image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                      floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                      floor((float(h_zoomed) / 2) + (float(h) / 2))))
            return image

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
