from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence
import time

import numpy as np
import rerun as rr


PoseEdgeType = Literal["odometry"]


@dataclass
class PoseNode:
    node_id: int
    x: float
    y: float
    theta: float


@dataclass
class PoseEdge:
    source: int
    target: int
    edge_type: PoseEdgeType


class PoseGraph:
    """Minimal pose-graph container with rerun logging helpers and database storage."""

    def __init__(self,
                 min_translation: float = 0.02,
                 min_rotation: float = np.deg2rad(1.0),
                 db_path: Optional[str] = None,
                 session_name: Optional[str] = None) -> None:
        self.min_translation = min_translation
        self.min_rotation = min_rotation
        self.nodes: List[PoseNode] = []
        self.edges: List[PoseEdge] = []
        self._dirty = False
        
        # Database integration
        self.db = None
        self.session_id = None
        if db_path:
            from .pose_graph_db import PoseGraphDB
            self.db = PoseGraphDB(db_path)
            if session_name:
                self.session_id = self.db.create_session(
                    session_name, 
                    f"Mapping session started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

    @staticmethod
    def _angle_diff(rad_a: float, rad_b: float) -> float:
        diff = rad_a - rad_b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def add_pose(self, x: float, y: float, theta: float) -> int:
        """Add a pose node if it is sufficiently different from the latest one."""
        if self.nodes:
            last = self.nodes[-1]
            pos_delta = np.hypot(last.x - x, last.y - y)
            rot_delta = abs(self._angle_diff(theta, last.theta))
            if pos_delta < self.min_translation and rot_delta < self.min_rotation:
                return last.node_id

        node_id = len(self.nodes)
        node = PoseNode(node_id, x, y, theta)
        self.nodes.append(node)
        
        # Save to database if available
        if self.db:
            self.db.add_node(node, self.session_id, time.time())
        
        if node_id > 0:
            self._add_edge(node_id - 1, node_id, "odometry")
        self._dirty = True
        return node_id


    def _add_edge(self, source: int, target: int, edge_type: PoseEdgeType) -> None:
        edge = PoseEdge(source, target, edge_type)
        self.edges.append(edge)
        
        # Save to database if available
        if self.db:
            self.db.add_edge(edge)
        
        self._dirty = True

    def log_to_rerun(self, one_map) -> None:
        """Log the current pose graph into rerun aligned with the map space."""
        if not self._dirty or not self.nodes:
            return

        node_pixels = self._nodes_to_pixels(one_map)
        colors = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (node_pixels.shape[0], 1))
        rr.log("map/pose_graph/nodes", rr.Points2D(node_pixels, colors=colors, radii=[0.5] * len(node_pixels)))

        self._log_edges(node_pixels)

        self._dirty = False

    def _nodes_to_pixels(self, one_map) -> np.ndarray:
        px_coords = []
        for node in self.nodes:
            px, py = one_map.metric_to_px(node.x, node.y)
            px_coords.append([py, px])
        return np.array(px_coords, dtype=np.float32)


    def _log_edges(self, node_pixels: np.ndarray) -> None:
        odom_edges: List[Sequence[Sequence[float]]] = []
        for edge in self.edges:
            segment = [node_pixels[edge.source], node_pixels[edge.target]]
            odom_edges.append(segment)

        if odom_edges:
            rr.log("map/pose_graph/edges/odometry",
                   rr.LineStrips2D(np.array(odom_edges, dtype=np.float32),
                                   colors=[[51, 153, 255]] * len(odom_edges)))

    def load_from_database(self, session_id: Optional[int] = None) -> None:
        """Load pose graph from database."""
        if not self.db:
            return
        
        if session_id:
            nodes = self.db.get_session_nodes(session_id)
        else:
            # Load all nodes if no session specified
            nodes = []
            # This would need a method to get all nodes
        
        self.nodes = nodes
        
        # Load edges for these nodes
        self.edges = []
        edges = self.db.get_edges_by_type("odometry")
        self.edges.extend(edges)
        
        self._dirty = True


    def get_trajectory_length(self) -> float:
        """Calculate total trajectory length from odometry edges."""
        total_length = 0.0
        for edge in self.edges:
            if edge.edge_type == "odometry" and edge.source < len(self.nodes) and edge.target < len(self.nodes):
                source_node = self.nodes[edge.source]
                target_node = self.nodes[edge.target]
                length = np.hypot(target_node.x - source_node.x, target_node.y - source_node.y)
                total_length += length
        return total_length

    def get_statistics(self) -> dict:
        """Get pose graph statistics."""
        stats = {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'odometry_edges': len(self.edges),
            'trajectory_length': self.get_trajectory_length()
        }
        
        if self.db:
            db_stats = self.db.get_graph_statistics()
            stats.update(db_stats)
        
        return stats

    def export_to_file(self, filepath: str) -> None:
        """Export pose graph to JSON file."""
        if self.db:
            self.db.export_to_json(filepath)
        else:
            # Fallback: export in-memory data
            import json
            data = {
                'nodes': [{'node_id': n.node_id, 'x': n.x, 'y': n.y, 'theta': n.theta} for n in self.nodes],
                'edges': [{'source': e.source, 'target': e.target, 'edge_type': e.edge_type} for e in self.edges],
                'statistics': self.get_statistics()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()

