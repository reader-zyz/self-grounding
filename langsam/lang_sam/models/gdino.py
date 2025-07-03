import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from lang_sam.models.utils import get_device_type

device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class GDINO:
    def __init__(self):
        self.build_model()

    # def build_model(self, ckpt_path: str | None = None):
    #     model_id = "IDEA-Research/grounding-dino-base"
    #     cache_dir = ckpt_path if ckpt_path else None
    #     self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    #     self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache_dir).to(
    #         DEVICE
    #     )
    def build_model(self, ckpt_path: str | None = None):
        self.processor = AutoProcessor.from_pretrained('weight', local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained('weight', local_files_only=True).to(
            DEVICE
        )

    # def predict(
    #     self,
    #     pil_images: list[Image.Image],
    #     text_prompt: list[str],
    #     box_threshold: float,
    #     text_threshold: float,
    # ) -> list[dict]:
    #     for i, prompt in enumerate(text_prompt):
    #         if prompt[-1] != ".":
    #             text_prompt[i] += "."
    #     inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(DEVICE)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     results = self.processor.post_process_grounded_object_detection(
    #         outputs,
    #         inputs.input_ids,
    #         box_threshold=box_threshold,
    #         text_threshold=text_threshold,
    #         target_sizes=[k.size[::-1] for k in pil_images],
    #     )
    #     return results

    def predict(
        self,
        pil_images: list[Image.Image],
        text_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        # Add period if missing (original code)
        for i, prompt in enumerate(text_prompt):
            if prompt[-1] != ".":
                text_prompt[i] += "."
        
        inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        
        # Process results to keep only highest confidence detection per label
        processed_results = []
        for image_result in results:
            # Create a dictionary to store the best detection for each label
            best_detections = {}
            
            for score, label, box in zip(image_result["scores"], image_result["labels"], image_result["boxes"]):
                # If label not in dict or current score is higher than stored one
                if label not in best_detections or score > best_detections[label]["score"]:
                    best_detections[label] = {
                        "score": score,
                        "box": box
                    }
            
            # Reconstruct the results in the original format
            if best_detections:
                # Sort by label to maintain consistent order (optional)
                sorted_labels = sorted(best_detections.keys())
                processed_result = {
                    "scores": torch.tensor([best_detections[label]["score"] for label in sorted_labels]),
                    "labels": sorted_labels,
                    "boxes": torch.stack([best_detections[label]["box"] for label in sorted_labels])
                }
            else:
                processed_result = {
                    "scores": torch.tensor([]),
                    "labels": [],
                    "boxes": torch.tensor([])
                }
            
            processed_results.append(processed_result)
        
        return processed_results


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
        ["wheel", "wheel"],
        0.3,
        0.25,
    )
    print(out)
