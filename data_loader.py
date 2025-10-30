# data_loader.py
import xml.etree.ElementTree as ET
import pandas as pd
from utils import DATA_DIR, get_image_paths
import random
from datetime import datetime, timedelta

def load_pklot_annotations():
    data = []
    xml_files = list(DATA_DIR.rglob("*.xml"))
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find('filename').text
        for space in root.findall('.//space'):
            occupied = 1 if space.get('occupied') == '1' else 0
            data.append({
                'image': img_name,
                'space_id': space.get('id'),
                'occupied': occupied,
                'contour': space.find('contour').attrib['points']
            })
    return pd.DataFrame(data)

def create_synthetic_logs(n=1000):
    plates = [f"R{f}{random.randint(100,999)}A" for f in "ABCD"]
    start = datetime(2025, 1, 1)
    logs = []
    for i in range(n):
        entry = start + timedelta(minutes=random.randint(0, 1440*30))
        duration = random.randint(10, 300)
        exit_time = entry + timedelta(minutes=duration)
        logs.append({
            'license_plate': random.choice(plates),
            'entry_time': entry,
            'exit_time': exit_time,
            'duration_min': duration,
            'parking_lot': random.choice(['Mall_A', 'Office_B', 'Residence_C']),
            'slot_id': f"S{random.randint(1, 200)}"
        })
    return pd.DataFrame(logs)

def load_all_data():
    print("Loading PKLot annotations...")
    pklot_df = load_pklot_annotations()
    print(f"Loaded {len(pklot_df)} parking space records.")

    print("Generating synthetic entry/exit logs...")
    logs_df = create_synthetic_logs(1500)
    print(f"Generated {len(logs_df)} log entries.")

    return pklot_df, logs_df