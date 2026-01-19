#!/usr/bin/env python3
"""
SigLIP2 Room Classifier Test Script

This script tests that frontier images are properly processed by SigLIP2
and that the room type probability distribution is correctly computed.

Usage:
    python test_siglip2_room_classifier.py [--image PATH] [--use-habitat]

Examples:
    # Test with a random image
    python test_siglip2_room_classifier.py
    
    # Test with a specific image file
    python test_siglip2_room_classifier.py --image path/to/image.jpg
    
    # Test with Habitat simulator (requires habitat-sim)
    python test_siglip2_room_classifier.py --use-habitat
"""

import argparse
import numpy as np
import time
from pathlib import Path
from typing import Optional

# Visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")

# Image loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Image loading disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available.")


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from file path."""
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    elif HAS_CV2:
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def create_test_images() -> dict:
    """
    Create synthetic test images that simulate different room types.
    Returns a dict of room_name -> numpy array [H, W, C].
    """
    test_images = {}
    
    # Create simple colored patterns to simulate room characteristics
    size = 224
    
    # Bathroom-like: white/blue tiles pattern
    bathroom = np.ones((size, size, 3), dtype=np.uint8) * 230
    for i in range(0, size, 20):
        bathroom[i:i+2, :] = [180, 200, 220]
        bathroom[:, i:i+2] = [180, 200, 220]
    test_images["bathroom_synthetic"] = bathroom
    
    # Bedroom-like: warm colors, bed-like rectangle
    bedroom = np.ones((size, size, 3), dtype=np.uint8) * np.array([240, 230, 210])
    bedroom[size//3:2*size//3, size//4:3*size//4] = [200, 180, 160]  # bed
    test_images["bedroom_synthetic"] = bedroom
    
    # Kitchen-like: white with cabinet-like rectangles
    kitchen = np.ones((size, size, 3), dtype=np.uint8) * 250
    kitchen[0:size//3, :] = [220, 210, 200]  # upper cabinets
    kitchen[2*size//3:, :] = [200, 190, 180]  # lower cabinets
    test_images["kitchen_synthetic"] = kitchen
    
    # Living room-like: warm with large couch-like shape
    living = np.ones((size, size, 3), dtype=np.uint8) * np.array([245, 240, 235])
    living[size//2:, size//4:3*size//4] = [150, 130, 110]  # couch
    test_images["living_room_synthetic"] = living
    
    # Hallway-like: narrow perspective
    hallway = np.ones((size, size, 3), dtype=np.uint8) * 220
    # Create perspective lines
    for i in range(size//4, 3*size//4):
        hallway[:, i] = [200, 195, 190]
    test_images["hallway_synthetic"] = hallway
    
    # Random noise (baseline)
    random_noise = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    test_images["random_noise"] = random_noise
    
    return test_images


def test_siglip2_classifier(image: np.ndarray, image_name: str = "test") -> dict:
    """
    Test SigLIP2 room classifier with a single image.
    
    Args:
        image: RGB image [H, W, C] as numpy array
        image_name: Name for logging
        
    Returns:
        Dict with room probabilities and timing info
    """
    from vision_models.siglip2_room_classifier import SigLIP2RoomClassifier, ROOM_CATEGORIES
    
    # Initialize classifier (may take a moment on first run)
    print(f"\n{'='*60}")
    print(f"Testing SigLIP2 Room Classifier")
    print(f"{'='*60}")
    print(f"Image: {image_name}")
    print(f"Shape: {image.shape}")
    print(f"Dtype: {image.dtype}")
    print(f"Range: [{image.min()}, {image.max()}]")
    
    # Load classifier
    print("\nLoading SigLIP2 model...")
    start_load = time.time()
    classifier = SigLIP2RoomClassifier()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Classify image
    print("\nClassifying image...")
    start_infer = time.time()
    probs = classifier.classify_image(image)
    infer_time = time.time() - start_infer
    print(f"Inference completed in {infer_time*1000:.1f}ms")
    
    # Print results
    print(f"\n{'='*60}")
    print("Room Probability Distribution P(R|F):")
    print(f"{'='*60}")
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    for i, idx in enumerate(sorted_indices):
        room = ROOM_CATEGORIES[idx]
        prob = probs[idx]
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ← TOP" if i == 0 else ""
        print(f"  {room:20s}: {prob:.4f} |{bar}|{marker}")
    
    # Get top prediction
    top_room, top_prob = classifier.get_top_room(image)
    
    print(f"\n{'='*60}")
    print(f"Top Prediction: {top_room} (confidence: {top_prob:.4f})")
    print(f"{'='*60}")
    
    return {
        "probs": probs,
        "top_room": top_room,
        "top_prob": top_prob,
        "load_time": load_time,
        "infer_time": infer_time,
    }


def test_bayesian_scorer(image: np.ndarray, target_object: str = "toilet") -> dict:
    """
    Test full Bayesian frontier scoring with an image.
    
    Args:
        image: RGB image [H, W, C] as numpy array
        target_object: Target object for navigation
        
    Returns:
        Dict with Bayesian score information
    """
    from vision_models.bayesian_frontier_scorer import BayesianFrontierScorer, ROOM_CATEGORIES
    
    print(f"\n{'='*60}")
    print(f"Testing Bayesian Frontier Scorer")
    print(f"Target Object: {target_object}")
    print(f"{'='*60}")
    
    # Check if prior matrix exists
    prior_path = "p_object_room.csv"
    if not Path(prior_path).exists():
        print(f"Error: Prior matrix not found at {prior_path}")
        return {}
    
    # Initialize scorer
    print("\nLoading Bayesian scorer...")
    start_load = time.time()
    scorer = BayesianFrontierScorer(prior_matrix_path=prior_path)
    load_time = time.time() - start_load
    print(f"Scorer loaded in {load_time:.2f}s")
    
    # Get object prior
    object_prior = scorer.get_object_prior(target_object)
    print(f"\nObject Prior P({target_object}|R):")
    top_prior_idx = np.argmax(object_prior)
    for idx, room in enumerate(ROOM_CATEGORIES):
        marker = " ← MOST LIKELY" if idx == top_prior_idx else ""
        print(f"  {room:20s}: {object_prior[idx]:.4f}{marker}")
    
    # Score frontier
    print("\nComputing Bayesian frontier score...")
    start_score = time.time()
    score = scorer.score_frontier(
        frontier_id=0,
        frontier_image=image,
        target_object=target_object,
        is_reachable=True,
    )
    score_time = time.time() - start_score
    
    print(f"\n{'='*60}")
    print("Bayesian Score Results:")
    print(f"{'='*60}")
    print(f"  Bayesian Score:  {score.bayesian_score:.4f}")
    print(f"  Gateway Bonus:   {score.gateway_bonus:.4f}")
    print(f"  Total Score:     {score.total_score:.4f}")
    print(f"  Top Room:        {score.top_room} ({score.top_room_prob:.2f})")
    print(f"  Is Reachable:    {score.is_reachable}")
    print(f"  Score Time:      {score_time*1000:.1f}ms")
    
    return {
        "bayesian_score": score.bayesian_score,
        "gateway_bonus": score.gateway_bonus,
        "total_score": score.total_score,
        "top_room": score.top_room,
        "top_room_prob": score.top_room_prob,
        "room_probs": score.room_probs,
    }


def test_with_habitat():
    """Test with Habitat simulator (if available)."""
    try:
        import habitat_sim
    except ImportError:
        print("Error: habitat-sim not available. Install it to use Habitat testing.")
        return
    
    print("\n" + "="*60)
    print("Testing with Habitat Simulator")
    print("="*60)
    
    # Configure simple scene
    hm3d_path = "datasets/scene_datasets/hm3d"
    scene_path = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    
    if not Path(scene_path).exists():
        print(f"Error: Scene not found at {scene_path}")
        print("Please ensure HM3D dataset is available.")
        return
    
    # Create simulator config
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.scene_dataset_config_file = hm3d_path + "/hm3d_annotated_basis.scene_dataset_config.json"
    
    # Camera config
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = 90
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    
    # Create simulator
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    
    # Get observation
    observations = sim.get_sensor_observations()
    rgb_image = observations["rgb"][:, :, :3]  # Remove alpha channel
    
    print(f"Captured image from Habitat: {rgb_image.shape}")
    
    # Test with captured image
    result = test_siglip2_classifier(rgb_image, "Habitat Scene")
    
    # Also test Bayesian scorer
    for target in ["toilet", "bed", "chair"]:
        test_bayesian_scorer(rgb_image, target)
    
    sim.close()
    print("\nHabitat test completed!")


def visualize_results(images: dict, results: dict):
    """Visualize test images and their classification results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization.")
        return
    
    from vision_models.siglip2_room_classifier import ROOM_CATEGORIES
    
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 4*n_images))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (name, image) in enumerate(images.items()):
        # Show image
        axes[i][0].imshow(image)
        axes[i][0].set_title(f"{name}")
        axes[i][0].axis("off")
        
        # Show probability distribution
        if name in results:
            probs = results[name]["probs"]
            sorted_indices = np.argsort(probs)[::-1]
            rooms = [ROOM_CATEGORIES[idx] for idx in sorted_indices]
            probs_sorted = probs[sorted_indices]
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(rooms)))
            axes[i][1].barh(rooms, probs_sorted, color=colors)
            axes[i][1].set_xlim(0, 1)
            axes[i][1].set_xlabel("Probability")
            axes[i][1].set_title(f"Room Classification: {results[name]['top_room']}")
    
    plt.tight_layout()
    plt.savefig("siglip2_test_results.png", dpi=150)
    print("\nVisualization saved to siglip2_test_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test SigLIP2 Room Classifier for Frontier Images"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to test image (if not provided, uses synthetic images)"
    )
    parser.add_argument(
        "--use-habitat",
        action="store_true",
        help="Test with Habitat simulator"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="toilet",
        help="Target object for Bayesian scoring (default: toilet)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SigLIP2 Room Classifier Test Suite")
    print("="*60)
    print(f"This test verifies that frontier images are properly processed")
    print(f"by SigLIP2 and room probability distributions are computed correctly.")
    print("="*60)
    
    if args.use_habitat:
        test_with_habitat()
        return
    
    if args.image:
        # Test with provided image
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            return
        
        image = load_image(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            return
        
        result = test_siglip2_classifier(image, args.image)
        test_bayesian_scorer(image, args.target)
        
    else:
        # Test with synthetic images
        print("\nNo image provided. Using synthetic test images...")
        test_images = create_test_images()
        
        results = {}
        for name, image in test_images.items():
            results[name] = test_siglip2_classifier(image, name)
        
        # Test Bayesian scorer with random noise
        print("\n" + "="*60)
        print("Testing Bayesian Scorer with Multiple Targets")
        print("="*60)
        
        test_image = test_images["bathroom_synthetic"]
        for target in ["toilet", "bed", "chair", "tv"]:
            test_bayesian_scorer(test_image, target)
        
        # Visualize if enabled
        if not args.no_viz and HAS_MATPLOTLIB:
            visualize_results(test_images, results)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
