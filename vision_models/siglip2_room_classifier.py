"""
SigLIP2-based Room Type Classifier for Bayesian Frontier Selection.

Uses HuggingFace's transformers library to load SigLIP2 model and classify
frontier images into room categories for ZSON (Zero-Shot Object Navigation).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image

# HuggingFace transformers
from transformers import AutoProcessor, AutoModel


# Room categories for HM3DSEM dataset (10 categories)
ROOM_CATEGORIES = [
    "bathroom",
    "bedroom", 
    "dining_room",
    "garage",
    "hall_stairwell",
    "kitchen",
    "laundry_room",
    "living_room",
    "office",
    "rec_room"
]

# Text prompts for zero-shot room classification
ROOM_PROMPTS = [
    "a photo of a bathroom",
    "a photo of a bedroom",
    "a photo of a dining room",
    "a photo of a garage",
    "a photo of a hallway or stairwell",
    "a photo of a kitchen",
    "a photo of a laundry room",
    "a photo of a living room",
    "a photo of an office",
    "a photo of a recreation room"
]


class SigLIP2RoomClassifier:
    """
    SigLIP2-based room type classifier for frontier images.
    
    Uses zero-shot classification to predict room type probabilities
    from frontier observation images.
    """
    
    # List of models to try in order of preference (smaller models first for memory efficiency)
    MODEL_CANDIDATES = [
        "google/siglip-base-patch16-224",    # Base model (~400MB, memory efficient)
        "google/siglip-base-patch16-256",    # Medium quality
        "openai/clip-vit-base-patch32",      # Fallback to CLIP (~600MB)
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize SigLIP2 room classifier.
        
        Args:
            model_name: HuggingFace model name/path for SigLIP (auto-detect if None)
            device: Device to run model on ('cuda' or 'cpu')
            cache_dir: Optional cache directory for model weights
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Try loading models in order of preference
        models_to_try = [model_name] if model_name else self.MODEL_CANDIDATES
        
        self.model = None
        self.processor = None
        self.model_name = None
        
        for candidate in models_to_try:
            if candidate is None:
                continue
            try:
                print(f"[SigLIP] Trying to load model: {candidate}")
                self._load_model(candidate)
                if self.model is not None:
                    print(f"[SigLIP] Successfully loaded: {candidate}")
                    break
            except Exception as e:
                print(f"[SigLIP] Failed to load {candidate}: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError(
                "[SigLIP] Failed to load any vision-language model. "
                "Please install transformers>=4.35.0 and ensure network access."
            )
        
        # Pre-compute text embeddings for room categories
        self._precompute_text_embeddings()
        
        print(f"[SigLIP] Model loaded successfully. Text embeddings cached for {len(ROOM_CATEGORIES)} room types.")
    
    def _load_model(self, model_name: str) -> None:
        """
        Attempt to load a specific model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
        
        # Check if it's a CLIP model (different API)
        if "clip" in model_name.lower() and "siglip" not in model_name.lower():
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
            )
            self.model = CLIPModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
            ).to(self.device)
            self.model.eval()
            self.model_name = model_name
            self._is_clip = True
        else:
            # SigLIP model
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                use_fast=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
            ).to(self.device)
            self.model.eval()
            self.model_name = model_name
            self._is_clip = False
    
    def _precompute_text_embeddings(self) -> None:
        """Pre-compute and cache text embeddings for room category prompts."""
        with torch.no_grad():
            if self._is_clip:
                # CLIP uses different input format
                text_inputs = self.processor(
                    text=ROOM_PROMPTS,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                text_outputs = self.model.get_text_features(**text_inputs)
            else:
                # SigLIP model
                text_inputs = self.processor(
                    text=ROOM_PROMPTS,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                text_outputs = self.model.get_text_features(**text_inputs)
            
            # Normalize embeddings
            self.text_embeddings = F.normalize(text_outputs, dim=-1)
    
    def classify_image(
        self,
        image: np.ndarray,
        return_dict: bool = False,
    ) -> np.ndarray:
        """
        Classify a single image into room type probabilities.
        
        Args:
            image: Input image as numpy array [H, W, C] in RGB format (0-255)
            return_dict: If True, return dict mapping room names to probabilities
            
        Returns:
            If return_dict=False: numpy array of shape (10,) with room probabilities
            If return_dict=True: dict mapping room category names to probabilities
        """
        with torch.no_grad():
            # Convert numpy to PIL Image
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Process image based on model type
            if self._is_clip:
                image_inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt",
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
            else:
                image_inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt",
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
            
            image_embeddings = F.normalize(image_outputs, dim=-1)
            
            # Compute similarity with pre-computed text embeddings
            logits = torch.matmul(image_embeddings, self.text_embeddings.T)
            
            # Apply temperature scaling and softmax for probability distribution
            temperature = 1.0
            probs = F.softmax(logits / temperature, dim=-1)
            
            probs_np = probs.cpu().numpy().squeeze()
        
        if return_dict:
            return {room: float(prob) for room, prob in zip(ROOM_CATEGORIES, probs_np)}
        return probs_np
    
    def classify_batch(
        self,
        images: List[np.ndarray],
    ) -> np.ndarray:
        """
        Classify a batch of images into room type probabilities.
        
        Args:
            images: List of images as numpy arrays [H, W, C] in RGB format
            
        Returns:
            numpy array of shape (N, 10) with room probabilities for each image
        """
        if len(images) == 0:
            return np.array([])
        
        with torch.no_grad():
            # Convert numpy arrays to PIL Images
            pil_images = []
            for image in images:
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_images.append(Image.fromarray(image))
            
            # Process images in batch based on model type
            if self._is_clip:
                image_inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
            else:
                image_inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                ).to(self.device)
                image_outputs = self.model.get_image_features(**image_inputs)
            
            image_embeddings = F.normalize(image_outputs, dim=-1)
            
            # Compute similarity
            logits = torch.matmul(image_embeddings, self.text_embeddings.T)
            
            # Apply softmax for probabilities
            temperature = 1.0
            probs = F.softmax(logits / temperature, dim=-1)
            
            return probs.cpu().numpy()
    
    def get_room_categories(self) -> List[str]:
        """Return list of room category names."""
        return ROOM_CATEGORIES.copy()
    
    def get_top_room(
        self,
        image: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Get the most likely room type for an image.
        
        Args:
            image: Input image as numpy array [H, W, C] in RGB format
            
        Returns:
            Tuple of (room_category_name, probability)
        """
        probs = self.classify_image(image)
        top_idx = np.argmax(probs)
        return ROOM_CATEGORIES[top_idx], float(probs[top_idx])
    
    def compute_object_verification_score(
        self,
        image_crop: np.ndarray,
        detected_label: str,
    ) -> Optional[float]:
        """
        Verify that a detected label matches the image region using SigLIP cosine similarity.
        Replaces CLIP-based verification for VRAM efficiency.
        
        Args:
            image_crop: Cropped image region [H, W, C] in RGB format (0-255)
            detected_label: Label detected by YOLO (e.g., "toilet", "bed")
            
        Returns:
            Cosine similarity score (roughly in [-1, 1]) or None if unavailable
        """
        try:
            # Skip too small crops
            if image_crop.shape[0] < 10 or image_crop.shape[1] < 10:
                return None
            
            # Convert numpy to PIL Image
            if image_crop.dtype != np.uint8:
                if image_crop.max() <= 1.0:
                    image_crop = (image_crop * 255).astype(np.uint8)
                else:
                    image_crop = image_crop.astype(np.uint8)
            
            pil_image = Image.fromarray(image_crop)
            
            with torch.no_grad():
                # Process image
                if self._is_clip:
                    image_inputs = self.processor(
                        images=pil_image,
                        return_tensors="pt",
                    ).to(self.device)
                    image_outputs = self.model.get_image_features(**image_inputs)
                else:
                    image_inputs = self.processor(
                        images=pil_image,
                        return_tensors="pt",
                    ).to(self.device)
                    image_outputs = self.model.get_image_features(**image_inputs)
                
                image_embeddings = F.normalize(image_outputs, dim=-1)
                
                # Process text (object label)
                text_prompt = f"a photo of a {detected_label}"
                if self._is_clip:
                    text_inputs = self.processor(
                        text=[text_prompt],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.get_text_features(**text_inputs)
                else:
                    text_inputs = self.processor(
                        text=[text_prompt],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.get_text_features(**text_inputs)
                
                text_embeddings = F.normalize(text_outputs, dim=-1)
                
                # Compute cosine similarity
                similarity = torch.matmul(image_embeddings, text_embeddings.T)
                
                return float(similarity.squeeze().cpu().item())
                
        except Exception as e:
            print(f"[SigLIP Verification] Error: {e}")
            return None


    def compute_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute normalized text embedding for a single string.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized embedding as numpy array (D,) or None if error
        """
        try:
            with torch.no_grad():
                if self._is_clip:
                    text_inputs = self.processor(
                        text=[text],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.get_text_features(**text_inputs)
                else:
                    text_inputs = self.processor(
                        text=[text],
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.get_text_features(**text_inputs)
                
                embedding = F.normalize(text_outputs, dim=-1)
                return embedding.cpu().numpy().squeeze()
        except Exception as e:
            print(f"[SigLIP] Text embedding error: {e}")
            return None

    def find_best_match(
        self, 
        query: str, 
        candidates: List[str],
        return_score: bool = False
    ) -> Union[str, Tuple[str, float], None]:
        """
        Find the best matching candidate string for a query using semantic similarity.
        
        Args:
            query: Query string
            candidates: List of candidate strings
            return_score: If True, return tuple (best_match, score)
            
        Returns:
            Best matching string, or (string, score) tuple, or None if error/empty
        """
        if not candidates:
            return None
            
        query_embedding = self.compute_text_embedding(query)
        if query_embedding is None:
            return None
            
        best_score = -1.0
        best_match = None
        
        # Compute embeddings for candidates (could be batched for efficiency, but list is usually small)
        for candidate in candidates:
            cand_embedding = self.compute_text_embedding(candidate)
            if cand_embedding is not None:
                score = np.dot(query_embedding, cand_embedding)
                if score > best_score:
                    best_score = score
                    best_match = candidate
                    
        if return_score:
            return best_match, float(best_score)
        return best_match


if __name__ == "__main__":
    # Test the SigLIP2 room classifier
    import cv2
    
    print("=" * 60)
    print("SigLIP2 Room Classifier Test")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SigLIP2RoomClassifier()
    
    # Create a dummy test image (random noise)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Classify
    probs = classifier.classify_image(test_image, return_dict=True)
    
    print("\nRoom probability distribution (random image):")
    for room, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {room:20s}: {prob:.4f}")
    
    top_room, top_prob = classifier.get_top_room(test_image)
    print(f"\nTop prediction: {top_room} ({top_prob:.4f})")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
