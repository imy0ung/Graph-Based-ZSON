"""
Bayesian Frontier Scorer for Zero-Shot Object Navigation (ZSON).

Combines SigLIP2 visual room classification with spatial common-sense priors
(P(Object|Room)) to compute Bayesian frontier scores for object navigation.

Implements:
- Bayesian score: S(F) = Σ P(O_target|R_j) × P(R_j|F)
- Gateway Strategy: bonus for hall_stairwell when navigating to specific rooms
- Fallback logic: automatically select next-best frontier when top is unreachable
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from vision_models.siglip2_room_classifier import (
    SigLIP2RoomClassifier,
    ROOM_CATEGORIES,
)


@dataclass
class FrontierScore:
    """Data class for frontier scoring results."""
    frontier_id: int
    bayesian_score: float
    room_probs: np.ndarray  # P(R|F) from SigLIP2
    top_room: str  # argmax room
    top_room_prob: float
    gateway_bonus: float  # Additional bonus from Gateway Strategy
    distance_penalty: float  # Distance-based penalty (0.02 * meters)
    hysteresis_bonus: float  # Goal hysteresis bonus (+0.05 if previous target)
    total_score: float  # bayesian_score + gateway_bonus - distance_penalty + hysteresis_bonus
    is_reachable: bool  # Whether path planning succeeded


class BayesianFrontierScorer:
    """
    Bayesian frontier scorer combining visual room classification with 
    spatial common-sense priors for zero-shot object navigation.
    
    Score formula: S(F) = Σ P(O_target|R_j) × P(R_j|F)
    
    Where:
    - P(R_j|F): Room probability from SigLIP2 visual classifier
    - P(O_target|R_j): Prior probability from spatial common-sense matrix
    """
    
    # Target objects supported (6 categories from HM3DSEM)
    TARGET_OBJECTS = ["chair", "bed", "plant", "toilet", "sofa", "tv_monitor"]
    
    # Gateway strategy parameters
    GATEWAY_ALPHA = 0.25  # Transition weight for hall_stairwell bonus
    HALL_STAIRWELL_IDX = ROOM_CATEGORIES.index("hall_stairwell")
    
    # Anti-oscillation parameters
    DISTANCE_PENALTY_COEFF = 0.02  # Penalty per meter of path distance
    HYSTERESIS_BONUS = 0.10  # Increased bonus for maintaining previous goal
    GOAL_LOCK_MARGIN = 0.15  # New goal must exceed current by this margin to switch
    # 0.15 -> 0.0
    def __init__(
        self,
        prior_matrix_path: str = "p_object_room.csv",
        siglip_model_name: Optional[str] = None,  # Auto-detect best available model
        device: Optional[str] = None,
        lazy_load_siglip: bool = False,
    ):
        """
        Initialize Bayesian frontier scorer.
        
        Args:
            prior_matrix_path: Path to P(Object|Room) CSV file
            siglip_model_name: HuggingFace model name for SigLIP (None for auto-detect)
            device: Device to run SigLIP on
            lazy_load_siglip: If True, defer SigLIP loading until first use
        """
        self.prior_matrix_path = prior_matrix_path
        self.siglip_model_name = siglip_model_name
        self.device = device
        
        # Load P(Object|Room) prior matrix from CSV
        self.prior_matrix = self._load_prior_matrix(prior_matrix_path)
        
        # Initialize SigLIP2 room classifier
        self._siglip_classifier: Optional[SigLIP2RoomClassifier] = None
        if not lazy_load_siglip:
            self._init_siglip()
        
        print(f"[BayesianScorer] Initialized with {len(self.TARGET_OBJECTS)} target objects "
              f"and {len(ROOM_CATEGORIES)} room categories")
    
    def _init_siglip(self) -> None:
        """Initialize SigLIP2 classifier (lazy loading)."""
        if self._siglip_classifier is None:
            self._siglip_classifier = SigLIP2RoomClassifier(
                model_name=self.siglip_model_name,
                device=self.device,
            )
    
    @property
    def siglip_classifier(self) -> SigLIP2RoomClassifier:
        """Get SigLIP2 classifier (lazy loading)."""
        if self._siglip_classifier is None:
            self._init_siglip()
        return self._siglip_classifier
    
    def _load_prior_matrix(self, path: str) -> Dict[str, np.ndarray]:
        """
        Load P(Object|Room) prior matrix from CSV file.
        
        Expected CSV format:
        ,bathroom,bedroom,dining_room,garage,hall_stairwell,kitchen,laundry_room,living_room,office,rec_room
        chair,0.059,0.168,...
        bed,0.0,0.968,...
        ...
        
        Args:
            path: Path to CSV file
            
        Returns:
            Dict mapping object names to probability arrays (10 rooms)
        """
        prior_matrix = {}
        
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Prior matrix file not found: {path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # First row is header with room names
            
            # Verify room order matches our expected order
            csv_rooms = header[1:]  # Skip first column (object name)
            if csv_rooms != ROOM_CATEGORIES:
                print(f"[BayesianScorer] Warning: CSV room order differs from expected")
                print(f"  CSV: {csv_rooms}")
                print(f"  Expected: {ROOM_CATEGORIES}")
            
            # Read object rows
            for row in reader:
                if len(row) < 11:  # object_name + 10 room probs
                    continue
                object_name = row[0].strip().lower()
                probs = np.array([float(p) for p in row[1:]], dtype=np.float32)
                
                # Normalize to ensure probabilities sum to 1
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                
                prior_matrix[object_name] = probs
        
        print(f"[BayesianScorer] Loaded prior matrix for objects: {list(prior_matrix.keys())}")
        return prior_matrix
    
    def get_object_prior(self, target_object: str) -> np.ndarray:
        """
        Get P(Object|Room) prior vector for a target object.
        
        Args:
            target_object: Target object name (e.g., 'toilet', 'bed')
            
        Returns:
            numpy array of shape (10,) with P(O|R) for each room
        """
        # Normalize object name
        obj_key = target_object.strip().lower()
        
        # Handle synonyms
        synonym_map = {
            "tv": "tv_monitor",
            "television": "tv_monitor",
            "monitor": "tv_monitor",
            "couch": "sofa",
            "potted plant": "plant",
        }
        obj_key = synonym_map.get(obj_key, obj_key)
        
        if obj_key in self.prior_matrix:
            return self.prior_matrix[obj_key]
        
        # Fallback: use text similarity to find closest object in prior matrix
        if self.siglip_classifier is not None:
            candidates = list(self.prior_matrix.keys())
            match_result = self.siglip_classifier.find_best_match(obj_key, candidates, return_score=True)
            
            if match_result is not None:
                best_match, score = match_result
                # Threshold for accepting a match (e.g., 0.8)
                if score > 0.8:
                    print(f"[BayesianScorer] Fallback: '{target_object}' not found. "
                          f"Using prior for '{best_match}' (similarity: {score:.3f})")
                    # Cache the result for future use
                    self.prior_matrix[obj_key] = self.prior_matrix[best_match]
                    return self.prior_matrix[obj_key]
                else:
                    print(f"[BayesianScorer] Warning: '{target_object}' not found. "
                          f"Best match '{best_match}' score {score:.3f} too low. Using uniform.")
            else:
                print(f"[BayesianScorer] Warning: '{target_object}' not found and matching failed. Using uniform.")
        else:
            print(f"[BayesianScorer] Warning: Object '{target_object}' not in prior matrix. Using uniform.")
            
        return np.ones(len(ROOM_CATEGORIES), dtype=np.float32) / len(ROOM_CATEGORIES)
    
    def compute_room_probabilities(self, frontier_image: np.ndarray) -> np.ndarray:
        """
        Compute P(R|F) - room probability distribution for a frontier image.
        
        Args:
            frontier_image: RGB image [H, W, C] as numpy array
            
        Returns:
            numpy array of shape (10,) with room probabilities
        """
        return self.siglip_classifier.classify_image(frontier_image)
    
    def compute_bayesian_score(
        self,
        target_object: str,
        room_probs: np.ndarray,
    ) -> float:
        """
        Compute Bayesian score: S(F) = Σ P(O_target|R_j) × P(R_j|F)
        
        Args:
            target_object: Target object name
            room_probs: P(R|F) probability distribution from SigLIP2
            
        Returns:
            Bayesian score (float)
        """
        object_prior = self.get_object_prior(target_object)  # P(O|R)
        
        # Bayesian summation
        score = np.sum(object_prior * room_probs)
        return float(score)
    
    def compute_gateway_bonus(
        self,
        target_object: str,
        room_probs: np.ndarray,
    ) -> float:
        """
        Compute Gateway Strategy bonus for hall_stairwell frontiers.
        
        If argmax(P(R|F)) is hall_stairwell, add transition weight:
        bonus = α × P(O_target|R_target_room)
        
        where R_target_room is the room with highest P(O|R) for the target object.
        
        Args:
            target_object: Target object name
            room_probs: P(R|F) probability distribution
            
        Returns:
            Gateway bonus (float), 0 if not hall_stairwell
        """
        # Check if top room is hall_stairwell
        top_room_idx = np.argmax(room_probs)
        if top_room_idx != self.HALL_STAIRWELL_IDX:
            return 0.0
        
        # Get target room (room with highest P(O|R) for target object)
        object_prior = self.get_object_prior(target_object)
        target_room_idx = np.argmax(object_prior)
        target_room_prob = object_prior[target_room_idx]
        
        # Compute gateway bonus
        bonus = self.GATEWAY_ALPHA * target_room_prob
        return float(bonus)
    
    def score_frontier(
        self,
        frontier_id: int,
        frontier_image: np.ndarray,
        target_object: str,
        is_reachable: bool = True,
        path_distance_meters: float = 0.0,
        is_previous_target: bool = False,
    ) -> FrontierScore:
        """
        Compute full Bayesian score for a single frontier with anti-oscillation features.
        
        Args:
            frontier_id: Unique identifier for the frontier
            frontier_image: RGB image [H, W, C] from frontier viewpoint
            target_object: Target object to navigate to
            is_reachable: Whether path planning to this frontier succeeded
            path_distance_meters: Path distance to frontier in meters (for distance penalty)
            is_previous_target: Whether this frontier was the previous navigation target
            
        Returns:
            FrontierScore dataclass with all scoring information
        """
        # Get room probabilities from SigLIP2
        room_probs = self.compute_room_probabilities(frontier_image)
        
        # Compute Bayesian score
        bayesian_score = self.compute_bayesian_score(target_object, room_probs)
        
        # Compute Gateway bonus
        gateway_bonus = self.compute_gateway_bonus(target_object, room_probs)
        
        # Compute distance penalty (Anti-oscillation: prefer closer frontiers)
        distance_penalty = self.DISTANCE_PENALTY_COEFF * path_distance_meters
        
        # Compute hysteresis bonus (Anti-oscillation: prefer maintaining current goal)
        hysteresis_bonus = self.HYSTERESIS_BONUS if is_previous_target else 0.0
        
        # Total score with anti-oscillation adjustments
        total_score = bayesian_score + gateway_bonus - distance_penalty + hysteresis_bonus
        
        # Get top room
        top_room_idx = np.argmax(room_probs)
        top_room = ROOM_CATEGORIES[top_room_idx]
        top_room_prob = float(room_probs[top_room_idx])
        
        return FrontierScore(
            frontier_id=frontier_id,
            bayesian_score=bayesian_score,
            room_probs=room_probs,
            top_room=top_room,
            top_room_prob=top_room_prob,
            gateway_bonus=gateway_bonus,
            distance_penalty=distance_penalty,
            hysteresis_bonus=hysteresis_bonus,
            total_score=total_score,
            is_reachable=is_reachable,
        )
    
    def score_frontiers_batch(
        self,
        frontier_images: List[np.ndarray],
        target_object: str,
        reachability_mask: Optional[List[bool]] = None,
        path_distances_meters: Optional[List[float]] = None,
        previous_target_idx: Optional[int] = None,
    ) -> List[FrontierScore]:
        """
        Score multiple frontiers in batch for efficiency with anti-oscillation features.
        
        Args:
            frontier_images: List of RGB images [H, W, C]
            target_object: Target object to navigate to
            reachability_mask: Optional list of reachability flags
            path_distances_meters: Optional list of path distances in meters
            previous_target_idx: Index of the frontier that was the previous target (for hysteresis)
            
        Returns:
            List of FrontierScore for each frontier
        """
        if len(frontier_images) == 0:
            return []
        
        if reachability_mask is None:
            reachability_mask = [True] * len(frontier_images)
        
        if path_distances_meters is None:
            path_distances_meters = [0.0] * len(frontier_images)
        
        # Batch classify room probabilities
        all_room_probs = self.siglip_classifier.classify_batch(frontier_images)
        
        # Get object prior once
        object_prior = self.get_object_prior(target_object)
        
        results = []
        for i, (room_probs, is_reachable, path_dist) in enumerate(
            zip(all_room_probs, reachability_mask, path_distances_meters)
        ):
            # Compute Bayesian score
            bayesian_score = float(np.sum(object_prior * room_probs))
            
            # Compute Gateway bonus
            top_room_idx = np.argmax(room_probs)
            if top_room_idx == self.HALL_STAIRWELL_IDX:
                target_room_idx = np.argmax(object_prior)
                gateway_bonus = float(self.GATEWAY_ALPHA * object_prior[target_room_idx])
            else:
                gateway_bonus = 0.0
            
            # Compute distance penalty (Anti-oscillation)
            distance_penalty = self.DISTANCE_PENALTY_COEFF * path_dist
            
            # Compute hysteresis bonus (Anti-oscillation)
            is_previous_target = (previous_target_idx is not None and i == previous_target_idx)
            hysteresis_bonus = self.HYSTERESIS_BONUS if is_previous_target else 0.0
            
            # Total score with anti-oscillation adjustments
            total_score = bayesian_score + gateway_bonus - distance_penalty + hysteresis_bonus
            
            results.append(FrontierScore(
                frontier_id=i,
                bayesian_score=bayesian_score,
                room_probs=room_probs,
                top_room=ROOM_CATEGORIES[top_room_idx],
                top_room_prob=float(room_probs[top_room_idx]),
                gateway_bonus=gateway_bonus,
                distance_penalty=distance_penalty,
                hysteresis_bonus=hysteresis_bonus,
                total_score=total_score,
                is_reachable=is_reachable,
            ))
        
        return results
    
    def select_best_frontier(
        self,
        frontier_scores: List[FrontierScore],
        fallback_count: int = 1,
    ) -> Tuple[Optional[FrontierScore], List[FrontierScore]]:
        """
        Select the best frontier with fallback logic.
        
        Returns the highest-scoring reachable frontier. If the top frontier
        is unreachable, automatically falls back to the next-best option.
        
        Args:
            frontier_scores: List of FrontierScore objects
            fallback_count: Number of fallback candidates to return
            
        Returns:
            Tuple of (best_frontier, fallback_candidates)
            best_frontier is None if no reachable frontiers exist
        """
        if len(frontier_scores) == 0:
            return None, []
        
        # Sort by total score (descending)
        sorted_scores = sorted(frontier_scores, key=lambda x: x.total_score, reverse=True)
        
        # Find best reachable frontier
        best_frontier = None
        fallbacks = []
        
        for score in sorted_scores:
            if score.is_reachable:
                if best_frontier is None:
                    best_frontier = score
                elif len(fallbacks) < fallback_count:
                    fallbacks.append(score)
                else:
                    break
        
        return best_frontier, fallbacks
    
    def debug_print_scores(
        self,
        frontier_scores: List[FrontierScore],
        target_object: str,
    ) -> None:
        """
        Print debug information for frontier scores.
        
        Args:
            frontier_scores: List of FrontierScore objects
            target_object: Target object name
        """
        print(f"\n{'='*80}")
        print(f"[Bayesian Frontier Scores] Target: {target_object}")
        print(f"{'='*80}")
        
        # Sort by total score
        sorted_scores = sorted(frontier_scores, key=lambda x: x.total_score, reverse=True)
        
        for i, fs in enumerate(sorted_scores):
            reachable_str = "✓" if fs.is_reachable else "✗"
            
            # Build score breakdown string
            breakdown_parts = [f"bayes={fs.bayesian_score:.3f}"]
            if fs.gateway_bonus > 0:
                breakdown_parts.append(f"+gate={fs.gateway_bonus:.3f}")
            if fs.distance_penalty > 0:
                breakdown_parts.append(f"-dist={fs.distance_penalty:.3f}")
            if fs.hysteresis_bonus > 0:
                breakdown_parts.append(f"+hyst={fs.hysteresis_bonus:.3f}")
            
            breakdown_str = ", ".join(breakdown_parts)
            
            print(f"  [{i+1}] F{fs.frontier_id}: "
                  f"total={fs.total_score:.4f} ({breakdown_str}) | "
                  f"room={fs.top_room}({fs.top_room_prob:.2f}) | "
                  f"{reachable_str}")
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test the Bayesian frontier scorer
    import os
    
    print("=" * 60)
    print("Bayesian Frontier Scorer Test")
    print("=" * 60)
    
    # Check if prior matrix exists
    prior_path = "p_object_room.csv"
    if not os.path.exists(prior_path):
        print(f"Error: Prior matrix file not found: {prior_path}")
        print("Please ensure p_object_room.csv is in the current directory.")
        exit(1)
    
    # Initialize scorer
    scorer = BayesianFrontierScorer(
        prior_matrix_path=prior_path,
        lazy_load_siglip=True,  # Lazy load to speed up test
    )
    
    # Test object priors
    print("\n[Test] Object priors P(O|R):")
    for obj in ["toilet", "bed", "chair", "tv"]:
        prior = scorer.get_object_prior(obj)
        top_room_idx = np.argmax(prior)
        print(f"  {obj:12s}: top_room={ROOM_CATEGORIES[top_room_idx]} (P={prior[top_room_idx]:.3f})")
    
    # Test Bayesian scoring (without loading SigLIP2)
    print("\n[Test] Bayesian score computation (simulated room probs):")
    
    # Simulate room probabilities for a bathroom-looking frontier
    bathroom_probs = np.array([0.6, 0.05, 0.02, 0.01, 0.1, 0.1, 0.02, 0.05, 0.03, 0.02])
    
    # Score for toilet (should be high)
    toilet_score = scorer.compute_bayesian_score("toilet", bathroom_probs)
    print(f"  Bathroom frontier + toilet target: score={toilet_score:.4f}")
    
    # Score for bed (should be low)
    bed_score = scorer.compute_bayesian_score("bed", bathroom_probs)
    print(f"  Bathroom frontier + bed target: score={bed_score:.4f}")
    
    # Simulate hallway frontier
    hallway_probs = np.array([0.05, 0.05, 0.02, 0.01, 0.7, 0.05, 0.02, 0.05, 0.03, 0.02])
    
    # Gateway bonus test
    gateway_toilet = scorer.compute_gateway_bonus("toilet", hallway_probs)
    gateway_bed = scorer.compute_gateway_bonus("bed", hallway_probs)
    print(f"\n  Hallway frontier gateway bonus (toilet): {gateway_toilet:.4f}")
    print(f"  Hallway frontier gateway bonus (bed): {gateway_bed:.4f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
