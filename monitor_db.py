#!/usr/bin/env python3
"""
실시간으로 pose graph 데이터베이스를 모니터링하는 스크립트.
주기적으로 DB를 읽어서 노드/엣지 통계를 출력합니다.
"""
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime

def get_db_stats(db_path: str):
    """데이터베이스 통계를 가져옵니다."""
    if not Path(db_path).exists():
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    stats = {}
    
    # Pose nodes
    cursor.execute("SELECT COUNT(*) as count FROM pose_nodes")
    stats['pose_nodes'] = cursor.fetchone()[0]
    
    # Object nodes
    try:
        cursor.execute("SELECT COUNT(*) as count FROM object_nodes")
        stats['object_nodes'] = cursor.fetchone()[0]
        
        # Object nodes by category (label)
        cursor.execute("""
            SELECT label, COUNT(*) as count 
            FROM object_nodes 
            GROUP BY label
            ORDER BY count DESC
        """)
        stats['object_categories'] = {row['label']: row['count'] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        # Table doesn't exist yet (old schema)
        stats['object_nodes'] = 0
        stats['object_categories'] = {}
    
    # Edges
    cursor.execute("SELECT COUNT(*) as count FROM pose_edges")
    stats['edges'] = cursor.fetchone()[0]
    
    # Edge types
    cursor.execute("""
        SELECT edge_type, COUNT(*) as count 
        FROM pose_edges 
        GROUP BY edge_type
    """)
    stats['edge_types'] = {row['edge_type']: row['count'] for row in cursor.fetchall()}
    
    # Latest node
    cursor.execute("SELECT MAX(node_id) as max_id FROM pose_nodes")
    stats['latest_node_id'] = cursor.fetchone()[0] or 0
    
    # Latest timestamp
    cursor.execute("SELECT MAX(created_at) as latest FROM pose_nodes")
    stats['latest_timestamp'] = cursor.fetchone()[0]
    
    conn.close()
    return stats

def monitor_db(db_path: str = "pose_graph.db", interval: float = 1.0):
    """
    데이터베이스를 주기적으로 모니터링합니다.
    
    Args:
        db_path: 데이터베이스 파일 경로
        interval: 업데이트 간격 (초)
    """
    print(f"Monitoring database: {db_path}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    last_stats = None
    
    try:
        while True:
            # Clear screen (ANSI escape code)
            print("\033[2J\033[H", end="")
            
            print("=" * 70)
            print(f"POSE GRAPH DATABASE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            stats = get_db_stats(db_path)
            
            if stats is None:
                print(f"\n⚠️  Database file not found: {db_path}")
                print("Waiting for database to be created...")
            else:
                # Show statistics
                print(f"\n📊 STATISTICS")
                print(f"  Pose Nodes:     {stats['pose_nodes']:>6}")
                print(f"  Object Nodes:  {stats['object_nodes']:>6}")
                print(f"  Total Edges:   {stats['edges']:>6}")
                print(f"  Latest Node ID: {stats['latest_node_id']}")
                
                if stats.get('object_categories'):
                    print(f"\n🏷️  OBJECT CATEGORIES")
                    for label, count in stats['object_categories'].items():
                        print(f"  {label:20s}: {count:>6}")
                
                if stats['edge_types']:
                    print(f"\n📈 EDGE TYPES")
                    for edge_type, count in sorted(stats['edge_types'].items()):
                        print(f"  {edge_type:20s}: {count:>6}")
                
                # Show changes
                if last_stats is not None:
                    print(f"\n🔄 CHANGES (since last update)")
                    pose_delta = stats['pose_nodes'] - last_stats['pose_nodes']
                    edge_delta = stats['edges'] - last_stats['edges']
                    obj_delta = stats['object_nodes'] - last_stats['object_nodes']
                    
                    if pose_delta != 0:
                        print(f"  Pose Nodes:    {pose_delta:>+6}")
                    if edge_delta != 0:
                        print(f"  Edges:         {edge_delta:>+6}")
                    if obj_delta != 0:
                        print(f"  Object Nodes:  {obj_delta:>+6}")
                    
                    # Object category changes
                    if stats.get('object_categories') and last_stats.get('object_categories'):
                        all_categories = set(list(stats['object_categories'].keys()) + list(last_stats['object_categories'].keys()))
                        category_changes = False
                        for category in all_categories:
                            old_count = last_stats['object_categories'].get(category, 0)
                            new_count = stats['object_categories'].get(category, 0)
                            if new_count != old_count:
                                if not category_changes:
                                    print(f"  Categories:")
                                    category_changes = True
                                print(f"    {category:20s}: {new_count - old_count:>+6}")
                    
                    # Edge type changes
                    for edge_type in set(list(stats['edge_types'].keys()) + list(last_stats['edge_types'].keys())):
                        old_count = last_stats['edge_types'].get(edge_type, 0)
                        new_count = stats['edge_types'].get(edge_type, 0)
                        if new_count != old_count:
                            print(f"  {edge_type:20s}: {new_count - old_count:>+6}")
                
                last_stats = stats.copy()
            
            print("\n" + "=" * 70)
            print(f"Next update in {interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor pose graph database in real-time")
    parser.add_argument("--db", type=str, default="pose_graph.db", help="Database file path")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor_db(args.db, args.interval)

