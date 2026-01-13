import os
import numpy as np
import torch
from vision_models.bayesian_frontier_scorer import BayesianFrontierScorer, ROOM_CATEGORIES

def test_fallback():
    print("=" * 60)
    print("Testing Bayesian Frontier Scorer Fallback Logic")
    print("=" * 60)
    
    # Ensure p_object_room.csv exists
    if not os.path.exists("p_object_room.csv"):
        print("Creating dummy p_object_room.csv for testing...")
        with open("p_object_room.csv", "w") as f:
            header = "," + ",".join(ROOM_CATEGORIES)
            f.write(header + "\n")
            # Add 'chair' with specific distribution (high in living_room)
            chair_probs = [0.05] * 10
            chair_probs[ROOM_CATEGORIES.index("living_room")] = 0.55
            f.write("chair," + ",".join(map(str, chair_probs)) + "\n")
            
            # Add 'bed' with specific distribution (high in bedroom)
            bed_probs = [0.05] * 10
            bed_probs[ROOM_CATEGORIES.index("bedroom")] = 0.55
            f.write("bed," + ",".join(map(str, bed_probs)) + "\n")

    # Initialize scorer
    print("\nInitializing Scorer...")
    scorer = BayesianFrontierScorer(
        prior_matrix_path="p_object_room.csv",
        lazy_load_siglip=False, # Load immediately to test text embeddings
    )
    
    # Test Case 1: Existing object
    print("\n[Test 1] Requesting 'chair' (exists in prior matrix)")
    prior_chair = scorer.get_object_prior("chair")
    top_room = ROOM_CATEGORIES[np.argmax(prior_chair)]
    print(f"  Top room: {top_room} (Expected: living_room)")
    
    # Test Case 2: Missing object with semantic match
    print("\n[Test 2] Requesting 'armchair' (missing, should match 'chair')")
    prior_armchair = scorer.get_object_prior("armchair")
    top_room = ROOM_CATEGORIES[np.argmax(prior_armchair)]
    print(f"  Top room: {top_room} (Expected: living_room)")
    
    # Test Case 3: Missing object with semantic match
    print("\n[Test 3] Requesting 'cot' (missing, should match 'bed')")
    prior_cot = scorer.get_object_prior("cot")
    top_room = ROOM_CATEGORIES[np.argmax(prior_cot)]
    print(f"  Top room: {top_room} (Expected: bedroom)")
    
    # Test Case 4: Random string (should fail and return uniform)
    print("\n[Test 4] Requesting 'xyz123random' (should fail)")
    prior_random = scorer.get_object_prior("xyz123random")
    is_uniform = np.allclose(prior_random, 1.0/len(ROOM_CATEGORIES))
    print(f"  Is uniform: {is_uniform} (Expected: True)")

    # Test Case 5: 'potted_plant' (underscore)
    print("\n[Test 5] Requesting 'potted_plant'")
    # We need 'plant' in the dummy matrix for this to work as expected if it falls back to 'plant'
    # Let's add 'plant' to the scorer's matrix manually for this test if it wasn't loaded from CSV
    if "plant" not in scorer.prior_matrix:
        scorer.prior_matrix["plant"] = np.ones(10, dtype=np.float32) / 10.0
        print("  (Added dummy 'plant' to prior matrix for testing)")
        
    prior_pp = scorer.get_object_prior("potted_plant")
    # Check if it used fallback or synonym
    # Note: 'potted_plant' is NOT in the default synonym map (which has 'potted plant' with space)
    # So we expect it to trigger the Fallback mechanism and match 'plant'

    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_fallback()
