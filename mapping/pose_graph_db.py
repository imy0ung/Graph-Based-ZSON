"""
Pose Graph Database for persistent storage and efficient querying.
Uses SQLite for lightweight, local storage of pose nodes and edges.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import asdict
import numpy as np

from .pose_graph import PoseNode, Edge, EdgeKind, FrontierNode, RegionNode
# 호환성을 위해 Edge를 PoseEdge로도 사용
PoseEdge = Edge


class PoseGraphDB:
    """SQLite-based database for pose graph storage and management."""
    
    def __init__(self, db_path: str = "pose_graph.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for pose nodes and edges."""
        cursor = self.conn.cursor()
        
        # Pose nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pose_nodes (
                node_id INTEGER PRIMARY KEY,
                x REAL NOT NULL,
                y REAL NOT NULL,
                theta REAL NOT NULL,
                timestamp REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pose edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pose_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL,
                transform_matrix TEXT,
                covariance TEXT,
                confidence REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES pose_nodes (node_id),
                FOREIGN KEY (target_id) REFERENCES pose_nodes (node_id),
                UNIQUE(source_id, target_id, edge_type)
            )
        """)
        
        # Object nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS object_nodes (
                node_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                confidence REAL NOT NULL,
                num_observations INTEGER DEFAULT 1,
                last_seen_step INTEGER DEFAULT 0,
                embedding BLOB,
                position_covariance TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Frontier nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frontier_nodes (
                node_id TEXT PRIMARY KEY,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                is_explored INTEGER DEFAULT 0,
                semantic_hint TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Region nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS region_nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                embedding BLOB,
                member_object_ids TEXT,
                member_frontier_ids TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sessions table for managing different mapping sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT UNIQUE NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                description TEXT,
                metadata TEXT
            )
        """)
        
        # Link nodes to sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_nodes (
                session_id INTEGER,
                node_id INTEGER,
                PRIMARY KEY (session_id, node_id),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                FOREIGN KEY (node_id) REFERENCES pose_nodes (node_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_position ON pose_nodes (x, y)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON pose_edges (source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON pose_edges (target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON pose_edges (edge_type)")
        
        self.conn.commit()
    
    def add_node(self, node: PoseNode, session_id: Optional[int] = None, 
                 timestamp: Optional[float] = None, metadata: Optional[Dict] = None) -> int:
        """Add a pose node to the database."""
        cursor = self.conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO pose_nodes 
            (node_id, x, y, theta, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (node.node_id, node.x, node.y, node.theta, timestamp, metadata_json))
        
        # Link to session if provided
        if session_id is not None:
            cursor.execute("""
                INSERT OR IGNORE INTO session_nodes (session_id, node_id)
                VALUES (?, ?)
            """, (session_id, node.node_id))
        
        self.conn.commit()
        return node.node_id
    
    def add_edge(self, edge: PoseEdge, transform_matrix: Optional[np.ndarray] = None,
                 covariance: Optional[np.ndarray] = None, confidence: float = 1.0) -> int:
        """Add a pose edge to the database."""
        cursor = self.conn.cursor()
        
        transform_json = json.dumps(transform_matrix.tolist()) if transform_matrix is not None else None
        covariance_json = json.dumps(covariance.tolist()) if covariance is not None else None
        
        # Extract numeric IDs from string IDs
        # Note: Edge class uses 'src' and 'dst', not 'source' and 'target'
        # IDs are in format "prefix_hexstring" (e.g., "fr_ae273870")
        def str_to_numeric_id(node_id):
            if isinstance(node_id, int):
                return node_id
            if '_' in node_id:
                hex_part = node_id.split('_')[-1]
                try:
                    # Try to parse as integer first
                    return int(hex_part)
                except ValueError:
                    # If not a decimal integer, try as hex string
                    try:
                        return int(hex_part, 16)
                    except ValueError:
                        # Fallback to hash
                        return hash(node_id) % (2**31)
            else:
                return hash(node_id) % (2**31)
        
        source_id = str_to_numeric_id(edge.src)
        target_id = str_to_numeric_id(edge.dst)
        
        cursor.execute("""
            INSERT OR REPLACE INTO pose_edges 
            (source_id, target_id, edge_type, transform_matrix, covariance, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source_id, target_id, edge.kind, transform_json, covariance_json, confidence))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_object_node(self, obj_node, session_id: Optional[int] = None) -> str:
        """Add an object node to the database."""
        from .pose_graph import ObjectNode
        assert isinstance(obj_node, ObjectNode)
        
        cursor = self.conn.cursor()
        
        # Serialize embedding and covariance
        embedding_blob = None
        if obj_node.embedding is not None:
            embedding_blob = obj_node.embedding.tobytes()
        
        covariance_json = json.dumps(obj_node.position_covariance.tolist()) if obj_node.position_covariance is not None else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO object_nodes 
            (node_id, label, x, y, z, confidence, num_observations, last_seen_step, embedding, position_covariance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            obj_node.id,
            obj_node.label,
            float(obj_node.position[0]),
            float(obj_node.position[1]),
            float(obj_node.position[2]),
            float(obj_node.confidence),
            int(obj_node.num_observations),
            int(obj_node.last_seen_step),
            embedding_blob,
            covariance_json
        ))
        
        self.conn.commit()
        return obj_node.id
    
    def add_frontier_node(self, fr_node: FrontierNode, session_id: Optional[int] = None) -> str:
        """Add a frontier node to the database."""
        assert isinstance(fr_node, FrontierNode)
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO frontier_nodes 
            (node_id, x, y, z, is_explored, semantic_hint)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            fr_node.id,
            float(fr_node.position[0]),
            float(fr_node.position[1]),
            float(fr_node.position[2]),
            1 if fr_node.is_explored else 0,
            fr_node.semantic_hint
        ))
        
        self.conn.commit()
        return fr_node.id

    def add_region_node(self, region_node: RegionNode) -> str:
        """Add a region node to the database."""
        assert isinstance(region_node, RegionNode)

        cursor = self.conn.cursor()

        embedding_blob = None
        if region_node.embedding is not None:
            embedding_blob = region_node.embedding.tobytes()

        member_object_ids = json.dumps(region_node.member_object_ids)
        member_frontier_ids = json.dumps(region_node.member_frontier_ids)

        cursor.execute("""
            INSERT OR REPLACE INTO region_nodes
            (node_id, name, x, y, z, embedding, member_object_ids, member_frontier_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            region_node.id,
            region_node.name,
            float(region_node.center[0]),
            float(region_node.center[1]),
            float(region_node.center[2]),
            embedding_blob,
            member_object_ids,
            member_frontier_ids,
        ))

        self.conn.commit()
        return region_node.id
    
    def remove_frontier_node(self, frontier_id: str) -> None:
        """Remove a frontier node and its connected edges from the database."""
        cursor = self.conn.cursor()
        
        # Remove edges connected to this frontier node
        # Extract numeric ID from frontier_id using the same logic as add_edge
        # IDs are in format "prefix_hexstring" (e.g., "fr_ae273870")
        def str_to_numeric_id(node_id):
            if isinstance(node_id, int):
                return node_id
            if '_' in node_id:
                hex_part = node_id.split('_')[-1]
                try:
                    # Try to parse as integer first
                    return int(hex_part)
                except ValueError:
                    # If not a decimal integer, try as hex string
                    try:
                        return int(hex_part, 16)
                    except ValueError:
                        # Fallback to hash
                        return hash(node_id) % (2**31)
            else:
                return hash(node_id) % (2**31)
        
        try:
            numeric_id = str_to_numeric_id(frontier_id)
            
            # Remove pose_frontier edges where this frontier is the target
            # (pose_frontier edges: src=pose_id, dst=frontier_id)
            cursor.execute("""
                DELETE FROM pose_edges 
                WHERE edge_type = 'pose_frontier' 
                AND target_id = ?
            """, (numeric_id,))
        except Exception as e:
            # If we can't parse the ID, log but continue
            print(f"[DB] Warning: Could not parse frontier_id {frontier_id} for edge removal: {e}")
        
        # Remove the frontier node itself
        cursor.execute("DELETE FROM frontier_nodes WHERE node_id = ?", (frontier_id,))
        self.conn.commit()
    
    def get_node(self, node_id: int) -> Optional[PoseNode]:
        """Retrieve a pose node by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM pose_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        
        if row:
            return PoseNode(
                node_id=row['node_id'],
                x=row['x'],
                y=row['y'],
                theta=row['theta']
            )
        return None
    
    def get_nodes_in_range(self, center_x: float, center_y: float, radius: float) -> List[PoseNode]:
        """Get all nodes within a circular range."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM pose_nodes 
            WHERE (x - ?) * (x - ?) + (y - ?) * (y - ?) <= ? * ?
        """, (center_x, center_x, center_y, center_y, radius, radius))
        
        nodes = []
        for row in cursor.fetchall():
            nodes.append(PoseNode(
                node_id=row['node_id'],
                x=row['x'],
                y=row['y'],
                theta=row['theta']
            ))
        return nodes
    
    def get_edges_by_type(self, edge_type: EdgeKind) -> List[PoseEdge]:
        """Get all edges of a specific type."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM pose_edges WHERE edge_type = ?", (edge_type,))
        
        edges = []
        for row in cursor.fetchall():
            # Edge 생성 (호환성을 위해 id 생성)
            edge_id = f"e_{row['source_id']}_{row['target_id']}_{row['edge_type']}"
            src_id = f"pose_{row['source_id']}"
            dst_id = f"pose_{row['target_id']}"
            edges.append(Edge(
                id=edge_id,
                kind=row['edge_type'],
                src=src_id,
                dst=dst_id,
                T_rel=None,
                rel_pos=None
            ))
        return edges
    
    def get_node_neighbors(self, node_id: int) -> List[Tuple[PoseNode, PoseEdge]]:
        """Get all neighboring nodes and their connecting edges."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT n.*, e.edge_type, e.confidence
            FROM pose_nodes n
            JOIN pose_edges e ON (n.node_id = e.target_id OR n.node_id = e.source_id)
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.node_id != ?
        """, (node_id, node_id, node_id))
        
        neighbors = []
        for row in cursor.fetchall():
            node = PoseNode(
                node_id=row['node_id'],
                x=row['x'],
                y=row['y'],
                theta=row['theta']
            )
            # Edge 생성 (호환성을 위해 id 생성)
            source_id = min(node_id, row['node_id'])
            target_id = max(node_id, row['node_id'])
            edge_id = f"e_{source_id}_{target_id}_{row['edge_type']}"
            src_id = f"pose_{source_id}"
            dst_id = f"pose_{target_id}"
            edge = Edge(
                id=edge_id,
                kind=row['edge_type'],
                src=src_id,
                dst=dst_id,
                T_rel=None,
                rel_pos=None
            )
            neighbors.append((node, edge))
        return neighbors
    
    def create_session(self, session_name: str, description: str = "") -> int:
        """Create a new mapping session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_name, description)
            VALUES (?, ?)
        """, (session_name, description))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_session_nodes(self, session_id: int) -> List[PoseNode]:
        """Get all nodes belonging to a session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT n.* FROM pose_nodes n
            JOIN session_nodes sn ON n.node_id = sn.node_id
            WHERE sn.session_id = ?
            ORDER BY n.node_id
        """, (session_id,))
        
        nodes = []
        for row in cursor.fetchall():
            nodes.append(PoseNode(
                node_id=row['node_id'],
                x=row['x'],
                y=row['y'],
                theta=row['theta']
            ))
        return nodes
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pose graph."""
        cursor = self.conn.cursor()
        
        # Node count
        cursor.execute("SELECT COUNT(*) as count FROM pose_nodes")
        node_count = cursor.fetchone()['count']
        
        # Object node count
        cursor.execute("SELECT COUNT(*) as count FROM object_nodes")
        object_count = cursor.fetchone()['count']

        # Frontier node count
        cursor.execute("SELECT COUNT(*) as count FROM frontier_nodes")
        frontier_count = cursor.fetchone()['count']

        # Region node count
        cursor.execute("SELECT COUNT(*) as count FROM region_nodes")
        region_count = cursor.fetchone()['count']

        # Region categories
        cursor.execute("""
            SELECT name, COUNT(*) as count
            FROM region_nodes
            GROUP BY name
            ORDER BY count DESC
        """)
        region_categories = {row['name']: row['count'] for row in cursor.fetchall()}
        
        # Edge counts by type
        cursor.execute("""
            SELECT edge_type, COUNT(*) as count 
            FROM pose_edges 
            GROUP BY edge_type
        """)
        edge_counts = {row['edge_type']: row['count'] for row in cursor.fetchall()}
        
        # Trajectory length (sum of pose_pose edges)
        cursor.execute("""
            SELECT 
                SUM(SQRT(
                    (n2.x - n1.x) * (n2.x - n1.x) + 
                    (n2.y - n1.y) * (n2.y - n1.y)
                )) as total_distance
            FROM pose_edges e
            JOIN pose_nodes n1 ON e.source_id = n1.node_id
            JOIN pose_nodes n2 ON e.target_id = n2.node_id
            WHERE e.edge_type = 'pose_pose' OR e.edge_type = 'odometry'
        """)
        total_distance = cursor.fetchone()['total_distance'] or 0.0
        
        return {
            'node_count': node_count,
            'object_count': object_count,
            'frontier_count': frontier_count,
            'region_count': region_count,
            'region_categories': region_categories,
            'edge_counts': edge_counts,
            'total_distance': total_distance,
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
        }
    
    def export_to_json(self, output_path: str):
        """Export the entire pose graph to JSON format."""
        cursor = self.conn.cursor()
        
        # Get all nodes
        cursor.execute("SELECT * FROM pose_nodes ORDER BY node_id")
        nodes = [dict(row) for row in cursor.fetchall()]
        
        # Get all edges
        cursor.execute("SELECT * FROM pose_edges ORDER BY edge_id")
        edges = [dict(row) for row in cursor.fetchall()]
        
        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': self.get_graph_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
