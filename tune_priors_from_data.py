import pandas as pd
import numpy as np
import os

def tune_priors_from_data():
    # 1. Load HM3D data
    input_path = "HM3DSem-v0.2/Per_Category_Region_Per_Cat_Votes.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Read CSV, skipping the first row (header details) if necessary, but pandas handles headers well.
    # The file has a header on line 1.
    df = pd.read_csv(input_path)
    
    # Normalize column names to match p_object_room.csv
    # CSV Cols: Bathroom, Bedroom, Dining room, Garage, Hall/stairwell, Kitchen, Laundry room, Living room, Office, Rec room
    # Target Cols: bathroom, bedroom, dining_room, garage, hall_stairwell, kitchen, laundry_room, living_room, office, rec_room
    
    col_map = {
        'Bathroom': 'bathroom',
        'Bedroom': 'bedroom',
        'Dining room': 'dining_room',
        'Garage': 'garage',
        'Hall/stairwell': 'hall_stairwell',
        'Kitchen': 'kitchen',
        'Laundry room': 'laundry_room',
        'Living room': 'living_room',
        'Office': 'office',
        'Rec room': 'rec_room'
    }
    
    # Filter columns
    df_cols = ['Category Tag'] + list(col_map.keys())
    df = df[df_cols]
    df = df.rename(columns=col_map)
    df = df.set_index('Category Tag')
    
    # 2. Define Object Mappings (Aggregating similar categories)
    object_mappings = {
        'chair': ['chair', 'armchair', 'rocking chair', 'folding chair', 'dining chair', 'office chair', 'desk chair'],
        'bed': ['bed', 'bunk bed', 'double bed', 'single bed', 'kids bed'],
        'plant': ['plant', 'potted plant', 'decorative plant', 'flower', 'flowerpot'],
        'toilet': ['toilet'],
        'sofa': ['sofa', 'couch', 'l-shaped sofa', 'sofa chair', 'sofa seat', 'sofa set'],
        'tv_monitor': ['tv', 'monitor', 'led tv', 'wall tv', 'television', 'computer monitor']
    }
    
    # 3. Calculate Counts
    target_objects = list(object_mappings.keys())
    target_rooms = list(col_map.values())
    
    new_priors = pd.DataFrame(0.0, index=target_objects, columns=target_rooms)
    
    print("Aggregating counts...")
    for obj, categories in object_mappings.items():
        for cat in categories:
            if cat in df.index:
                # Sum counts for this category across all rooms
                counts = df.loc[cat]
                new_priors.loc[obj] += counts
                print(f"  Added '{cat}' to '{obj}'")
            else:
                # Try case-insensitive match
                matches = [idx for idx in df.index if str(idx).lower() == cat.lower()]
                for match in matches:
                    counts = df.loc[match]
                    new_priors.loc[obj] += counts
                    print(f"  Added '{match}' (match for '{cat}') to '{obj}'")

    print("\nRaw Counts:")
    print(new_priors)
    
    # 4. Normalize to get P(Room | Object)
    # Divide each row by its sum
    row_sums = new_priors.sum(axis=1)
    probs = new_priors.div(row_sums, axis=0)
    
    print("\nNormalized Probabilities (P(Room|Object)):")
    print(probs)
    
    # 5. Save
    output_path = "p_object_room.csv"
    # Backup existing if not already done (we did it before, but good practice)
    if os.path.exists(output_path) and not os.path.exists(output_path + ".bak_data"):
        os.rename(output_path, output_path + ".bak_data")
        
    probs.to_csv(output_path)
    print(f"\nSaved new priors to {output_path}")
    
    # 6. Compare with "Common Sense" / Previous Issues
    print("\nKey Stats:")
    print(f"Chair in Hallway: {probs.loc['chair', 'hall_stairwell']:.3f}")
    print(f"Chair in Dining:  {probs.loc['chair', 'dining_room']:.3f}")
    print(f"Plant in Hallway: {probs.loc['plant', 'hall_stairwell']:.3f}")
    print(f"Sofa in Hallway:  {probs.loc['sofa', 'hall_stairwell']:.3f}")

if __name__ == "__main__":
    tune_priors_from_data()
