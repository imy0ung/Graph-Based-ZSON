#!/usr/bin/env python3
"""
Check pose graph structure and database contents.
"""
import sqlite3
import json
from pathlib import Path
from mapping.pose_graph import PoseGraph

def check_database(db_path: str = "pose_graph.db"):
    """Check database contents."""
    if not Path(db_path).exists():
        print(f"Database file '{db_path}' not found.")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 60)
    print("DATABASE CONTENTS")
    print("=" * 60)
    
    # Tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nTables: {[t['name'] for t in tables]}")
    
    # Pose nodes
    cursor.execute("SELECT COUNT(*) as count FROM pose_nodes")
    node_count = cursor.fetchone()[0]
    print(f"\nPose Nodes: {node_count}")
    if node_count > 0:
        cursor.execute("SELECT node_id, x, y, theta FROM pose_nodes ORDER BY node_id LIMIT 10")
        print("  First 10 nodes:")
        for row in cursor.fetchall():
            node_id, x, y, theta = row
            try:
                print(f"    Node {node_id}: ({float(x):.3f}, {float(y):.3f}, {float(theta):.3f})")
            except (ValueError, TypeError):
                print(f"    Node {node_id}: ({x}, {y}, {theta})")
    
    # Pose edges
    cursor.execute("SELECT COUNT(*) as count FROM pose_edges")
    edge_count = cursor.fetchone()[0]
    print(f"\nPose Edges: {edge_count}")
    if edge_count > 0:
        cursor.execute("""
            SELECT edge_type, COUNT(*) as count 
            FROM pose_edges 
            GROUP BY edge_type
        """)
        print("  Edge types:")
        for row in cursor.fetchall():
            edge_type, count = row
            print(f"    {edge_type}: {count}")
        cursor.execute("SELECT source_id, target_id, edge_type FROM pose_edges ORDER BY edge_id LIMIT 10")
        print("  First 10 edges:")
        for row in cursor.fetchall():
            source_id, target_id, edge_type = row
            print(f"    {source_id} -> {target_id} ({edge_type})")
    
    # Sessions
    cursor.execute("SELECT COUNT(*) as count FROM sessions")
    session_count = cursor.fetchone()[0]
    print(f"\nSessions: {session_count}")
    if session_count > 0:
        cursor.execute("SELECT session_id, session_name, start_time FROM sessions ORDER BY session_id")
        for row in cursor.fetchall():
            session_id, session_name, start_time = row
            print(f"  Session {session_id}: {session_name} (started: {start_time})")
    
    conn.close()

def check_memory_graph(db_path: str = "pose_graph.db", session_id: int = None):
    """Check in-memory graph structure."""
    print("\n" + "=" * 60)
    print("IN-MEMORY GRAPH STRUCTURE")
    print("=" * 60)
    
    try:
        # Create graph without loading (to check current state)
        # Note: This won't load existing data, just shows structure
        pg = PoseGraph(db_path=None)  # Don't load from DB, just check structure
        
        # If you want to check a specific session, you can load it
        if db_path and Path(db_path).exists():
            pg_with_db = PoseGraph(db_path=db_path)
            if session_id:
                pg_with_db.load_from_database(session_id)
            pg = pg_with_db
        
        # Get statistics
        stats = pg.get_statistics()
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Node details
        print(f"\nPose Nodes: {len(pg.pose_ids)}")
        if pg.pose_ids:
            print("  First 5 pose nodes:")
            for i, pose_id in enumerate(pg.pose_ids[:5]):
                node = pg.nodes[pose_id]
                if hasattr(node, 'x'):
                    print(f"    {pose_id}: ({node.x:.3f}, {node.y:.3f}, {node.theta:.3f})")
        
        print(f"\nObject Nodes: {len(pg.object_ids)}")
        if pg.object_ids:
            print("  Object nodes:")
            # Group by label
            label_counts = {}
            for obj_id in pg.object_ids:
                obj = pg.nodes[obj_id]
                if hasattr(obj, 'label'):
                    label = obj.label
                    if label not in label_counts:
                        label_counts[label] = []
                    label_counts[label].append(obj)
            
            for label, objs in sorted(label_counts.items()):
                print(f"    {label}: {len(objs)} objects")
                for obj in objs[:3]:  # Show first 3 of each label
                    if hasattr(obj, 'num_observations') and hasattr(obj, 'confidence'):
                        print(f"      - {obj.id}: pos=({obj.position[0]:.3f}, {obj.position[1]:.3f}), "
                              f"obs={obj.num_observations}, conf={obj.confidence:.3f}")
        
        print(f"\nFrontier Nodes: {len(pg.frontier_ids)}")
        print(f"Region Nodes: {len(pg.region_ids)}")
        
        # Edge details
        print(f"\nEdges: {len(pg.edges)}")
        edge_types = {}
        for edge in pg.edges.values():
            edge_type = edge.kind
            if edge_type not in edge_types:
                edge_types[edge_type] = 0
            edge_types[edge_type] += 1
        
        print("  Edge types:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"    {edge_type}: {count}")
        
        # Sample edges
        print("\n  Sample edges (first 10):")
        for i, (edge_id, edge) in enumerate(list(pg.edges.items())[:10]):
            print(f"    {edge_id}: {edge.src} -> {edge.dst} ({edge.kind})")
        
        # Object-pose connections
        if pg.object_ids:
            print("\n  Object-Pose connections:")
            obj_pose_edges = [e for e in pg.edges.values() if e.kind == "pose_object"]
            print(f"    Total pose-object edges: {len(obj_pose_edges)}")
            if obj_pose_edges:
                # Group by object
                obj_connections = {}
                for edge in obj_pose_edges:
                    obj_id = edge.dst
                    if obj_id not in obj_connections:
                        obj_connections[obj_id] = []
                    obj_connections[obj_id].append(edge.src)
                
                print(f"    Objects with connections: {len(obj_connections)}")
                for obj_id, pose_ids in list(obj_connections.items())[:5]:
                    obj = pg.nodes[obj_id]
                    if hasattr(obj, 'label'):
                        print(f"      {obj.label} ({obj_id}): connected to {len(pose_ids)} pose(s)")
        
    except Exception as e:
        print(f"Error loading graph: {e}")
        import traceback
        traceback.print_exc()

def show_graph_structure(db_path: str = "pose_graph.db", max_nodes_per_type: int = 20):
    """Show graph structure as adjacency list."""
    print("\n" + "=" * 60)
    print("GRAPH STRUCTURE (Adjacency List)")
    print("=" * 60)
    
    try:
        pg = PoseGraph(db_path=db_path)
        
        if len(pg.nodes) == 0:
            print("Graph is empty.")
            return
        
        # Build adjacency list: node_id -> list of (neighbor_id, edge_type)
        adjacency = {}
        for edge in pg.edges.values():
            src = edge.src
            dst = edge.dst
            edge_type = edge.kind
            
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append((dst, edge_type))
            
            # For undirected edges, add reverse (if needed)
            # Most edges are directed, so we only add forward for now
        
        # Show pose nodes and their connections
        print(f"\n📍 POSE NODES ({len(pg.pose_ids)} nodes)")
        print("-" * 60)
        for i, pose_id in enumerate(pg.pose_ids[:max_nodes_per_type]):
            node = pg.nodes[pose_id]
            if hasattr(node, 'x'):
                print(f"\n{pose_id} [({node.x:.2f}, {node.y:.2f}), θ={node.theta:.2f}]")
            else:
                print(f"\n{pose_id}")
            
            if pose_id in adjacency:
                connections = adjacency[pose_id]
                for neighbor_id, edge_type in connections:
                    neighbor = pg.nodes.get(neighbor_id)
                    neighbor_info = ""
                    if neighbor:
                        if hasattr(neighbor, 'label'):
                            neighbor_info = f" [{neighbor.label}]"
                        elif hasattr(neighbor, 'x'):
                            neighbor_info = f" [({neighbor.x:.2f}, {neighbor.y:.2f})]"
                    print(f"  └─> {neighbor_id}{neighbor_info} ({edge_type})")
            else:
                print("  └─> (no outgoing connections)")
        
        if len(pg.pose_ids) > max_nodes_per_type:
            print(f"\n  ... and {len(pg.pose_ids) - max_nodes_per_type} more pose nodes")
        
        # Show object nodes and their connections
        if pg.object_ids:
            print(f"\n\n🏷️  OBJECT NODES ({len(pg.object_ids)} nodes)")
            print("-" * 60)
            
            # Group by label
            objects_by_label = {}
            for obj_id in pg.object_ids:
                obj = pg.nodes[obj_id]
                if hasattr(obj, 'label'):
                    label = obj.label
                    if label not in objects_by_label:
                        objects_by_label[label] = []
                    objects_by_label[label].append(obj_id)
            
            for label, obj_ids in sorted(objects_by_label.items()):
                print(f"\n  {label} ({len(obj_ids)} objects):")
                for obj_id in obj_ids[:max_nodes_per_type]:
                    obj = pg.nodes[obj_id]
                    print(f"    {obj_id} [pos=({obj.position[0]:.2f}, {obj.position[1]:.2f}), "
                          f"conf={obj.confidence:.2f}, obs={obj.num_observations}]")
                    
                    # Show which poses observe this object
                    incoming = []
                    for edge in pg.edges.values():
                        if edge.dst == obj_id and edge.kind == "pose_object":
                            incoming.append(edge.src)
                    
                    if incoming:
                        print(f"      └─ observed by: {', '.join(incoming[:5])}")
                        if len(incoming) > 5:
                            print(f"         ... and {len(incoming) - 5} more")
                
                if len(obj_ids) > max_nodes_per_type:
                    print(f"      ... and {len(obj_ids) - max_nodes_per_type} more {label} objects")
        
        # Show edge type summary
        print(f"\n\n📊 EDGE TYPE SUMMARY")
        print("-" * 60)
        edge_type_counts = {}
        for edge in pg.edges.values():
            edge_type = edge.kind
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        for edge_type, count in sorted(edge_type_counts.items()):
            print(f"  {edge_type:20s}: {count:>6} edges")
        
        # Show connectivity statistics
        print(f"\n\n📈 CONNECTIVITY STATISTICS")
        print("-" * 60)
        
        # Count degrees
        pose_degrees = {}
        for pose_id in pg.pose_ids:
            degree = len([e for e in pg.edges.values() if e.src == pose_id or e.dst == pose_id])
            pose_degrees[pose_id] = degree
        
        if pose_degrees:
            max_degree = max(pose_degrees.values())
            min_degree = min(pose_degrees.values())
            avg_degree = sum(pose_degrees.values()) / len(pose_degrees)
            print(f"  Pose nodes:")
            print(f"    Max degree: {max_degree}")
            print(f"    Min degree: {min_degree}")
            print(f"    Avg degree: {avg_degree:.2f}")
        
        # Object connectivity
        obj_connections = {}
        for edge in pg.edges.values():
            if edge.kind == "pose_object":
                obj_id = edge.dst
                obj_connections[obj_id] = obj_connections.get(obj_id, 0) + 1
        
        if obj_connections:
            max_obj_conn = max(obj_connections.values())
            min_obj_conn = min(obj_connections.values())
            avg_obj_conn = sum(obj_connections.values()) / len(obj_connections)
            print(f"  Object nodes:")
            print(f"    Max connections: {max_obj_conn}")
            print(f"    Min connections: {min_obj_conn}")
            print(f"    Avg connections: {avg_obj_conn:.2f}")
        
    except Exception as e:
        print(f"Error showing graph structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Check pose graph structure and database contents")
    parser.add_argument("db_path", nargs="?", default="pose_graph.db", help="Database file path")
    parser.add_argument("--structure", action="store_true", help="Show detailed graph structure")
    parser.add_argument("--max-nodes", type=int, default=20, help="Maximum nodes to show per type")
    
    args = parser.parse_args()
    
    check_database(args.db_path)
    check_memory_graph(args.db_path)
    
    if args.structure:
        show_graph_structure(args.db_path, max_nodes_per_type=args.max_nodes)

