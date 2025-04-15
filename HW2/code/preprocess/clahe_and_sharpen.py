"""
Contains the CLAHEandSharpen class for image augmentation.
"""
import random

from PIL import Image
import numpy as np
import cv2

class CLAHEandSharpen:
    """
    CLAHE and sharpening augmentation class.
    This class applies Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    and Laplacian sharpening to images.
    """
    def __init__(
        self,
        random_val=0,
        clip_limit=1.0,
        tile_grid_size=(8, 8),
        sigma=1.0,
        alpha=1.5,
        beta=-0.5,
    ):
        """
        Initialize the CLAHE and sharpening augmentation class.

        Args:
            random (float): Probability threshold for applying the augmentation.
            clip_limit (float): Threshold for contrast limiting in CLAHE.
            tile_grid_size (tuple): Size of the grid for histogram equalization in CLAHE.
            sigma (float): Standard deviation for Gaussian blur.
            alpha (float): Weight for the original image in sharpening.
            beta (float): Weight for the Laplacian image in sharpening.

        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.random_val = random_val
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img: Image.Image):
        """
        Apply CLAHE and sharpening to the input image with a certain probability.

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Augmented image.
        """
        img = np.array(img)
        if random.random() > self.random_val:
            # Convert from PIL (RGB) to OpenCV format (BGR)
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Apply CLAHE to the Y channel of the YUV color space
            img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = self.clahe.apply(img_yuv[:, :, 0])
            img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # Sharpening process:
            # Step 1: Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_clahe, (0, 0), self.sigma)

            # Step 2: Compute Laplacian (edge detection)
            laplacian = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
            laplacian = cv2.convertScaleAbs(laplacian)

            # Step 3: Combine original image and Laplacian (LoG sharpening)
            sharpened = cv2.addWeighted(img_clahe, self.alpha, laplacian, self.beta, 0)

            # Convert back from BGR to RGB (OpenCV to PIL format)
            img = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)


# # Example usage
# if __name__ == "__main__":
#     # Initialize the transform
#     transform = CLAHEandSharpen()

#     # Load an image
#     img_path = "../../nycu-hw2-data/train/"
#     for i in range(21, 30):
#         img_path2 = img_path + str(i) + ".png"
#         img = Image.open(img_path2)

#         # Apply the transformation
#         transformed_img = transform(img)

#         # Save or display the transformed image
#         transformed_img.save(f"transformed_{i}.png")
