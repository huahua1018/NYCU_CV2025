from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import random

class CLAHEandSharpen:
    def __init__(self, random=0, clip_limit=1.0, tile_grid_size=(8, 8), sigma=1.0, alpha=1.5, beta=-0.5):
        """
        clip_limit: CLAHE 對比限制
        tile_grid_size: CLAHE 分區格子數
        sigma: LoG 的高斯模糊強度
        alpha: 原圖的加權值
        beta: LoG 結果的加權值（通常為負值以實現銳化）
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.random = random
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img: Image.Image):
        img = np.array(img)
        if random.random() > self.random:
            # PIL to OpenCV format
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # #CLAHE
            # img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
            # img_yuv[:, :, 0] = self.clahe.apply(img_yuv[:, :, 0])
            # img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # 銳化
            # Gaussian Blur
            blurred = cv2.GaussianBlur(img_cv, (0, 0), self.sigma)

            # Laplacian
            laplacian = cv2.Laplacian(blurred, cv2.CV_16S, ksize=3)
            laplacian = cv2.convertScaleAbs(laplacian)

            # Sharpen: 原圖 + LoG
            sharpened = cv2.addWeighted(img_cv, self.alpha, laplacian, self.beta, 0)

            # 轉換回 RGB PIL.Image
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