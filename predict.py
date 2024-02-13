from cog import BasePredictor, Input, Path
from simple_lama_inpainting import SimpleLama
from PIL import Image


class Predictor(BasePredictor):
    def setup(self):
        self.simple_lama = SimpleLama()

    def predict(self,
                org_image: Path = Input(description="Original input image"),
                mask_image: Path = Input(description="Mask image")
                ) -> Path:
        """Run the inpainting process using SimpleLama"""

        # Load images using PIL
        original_image = Image.open(org_image)
        mask_image = Image.open(mask_image).convert('L')

        # Perform inpainting
        result = self.simple_lama(original_image, mask_image)
        result_path = "/tmp/in-painted.png"
        result.save(result_path)

        return Path(result_path)
