from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from semantic_prototypes import (
    PrototypeConfig,
    SemanticPrototypeIndex,
    annotate_object_nodes,
)
from vision_models.clip_dense import ClipModel


@dataclass
class MockObjectNode:
    label: str
    confidence: float


def run_demo() -> None:
    model = ClipModel("weights/clip.pth", jetson=False)
    config = PrototypeConfig(
        indoor_weight_min=0.05,
        indoor_weight_t_low=0.15,
        indoor_weight_t_high=0.30,
        min_room_sim_threshold=0.2,
        room_margin_threshold=0.05,
    )
    proto_index = SemanticPrototypeIndex(model, config=config, auto_build=False)
    proto_index.build_or_load(ignore_cache=True)
    proto_index.debug_dump(
        ["bed", "sofa", "airplane", "microwave", "toilet", "exit sign", "corridor", "hallway"]
    )

    print("Indoor weight mapping demo:")
    labels = [
        "bed",
        "sofa",
        "microwave",
        "toilet",
        "airplane",
        "train",
        "whale",
        "car",
        "tree",
        "street sign",
    ]
    for label in labels:
        info = proto_index.indoor_plausibility(label)
        conf_weighted = info["indoor_weight"] * 1.0
        print(
            f"  {label:10s} sim_in={info['sim_indoor']:.6f} "
            f"sim_out={info.get('sim_outdoor', 0.0):.6f} "
            f"margin={info.get('sim_margin', 0.0):.6f} "
            f"weight={info['indoor_weight']:.6f} conf_w={conf_weighted:.6f}"
        )

    print("\nRoom assignment demo:")
    labels = ["bed", "pillow", "fridge", "microwave", "toilet", "monitor", "printer", "exit sign"]
    for label in labels:
        info = proto_index.assign_room(label)
        print(
            f"  {label:10s} room={info['room']:12s} "
            f"score={info['score']:.6f} margin={info['margin']:.6f}"
        )

    print("\nObject node annotation demo:")
    nodes = [
        MockObjectNode("bed", 0.9),
        MockObjectNode("airplane", 0.8),
        MockObjectNode("microwave", 0.7),
    ]
    annotate_object_nodes(proto_index, nodes)
    for node in nodes:
        assert hasattr(node, "confidence_weighted")
        assert node.confidence_weighted == node.confidence * node.indoor_weight
        print(
            f"  {node.label:10s} conf={node.confidence:.2f} "
            f"weight={node.indoor_weight:.2f} conf_w={node.confidence_weighted:.2f} "
            f"room={node.room_type}"
        )


if __name__ == "__main__":
    run_demo()
