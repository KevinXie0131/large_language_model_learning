#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from datetime import datetime
from pymongo import MongoClient, InsertOne, ASCENDING

def to_int(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None

def to_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def to_bool(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None

def to_date(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    # Date formats matching: 2024-06-19T16:00:00.000Z or 2024-06-19
    fmts = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def drop_none_recursive(obj):
    if isinstance(obj, dict):
        return {
            k: drop_none_recursive(v) 
            for k, v in obj.items() 
            if v is not None and drop_none_recursive(v) is not None
        }
    elif isinstance(obj, list):
        new_list = [drop_none_recursive(v) for v in obj]
        new_list = [v for v in new_list if v is not None]
        return new_list if new_list else None
    else:
        return obj

def nest_dotted_keys(flat_dict):
    nested = {}
    for k, v in flat_dict.items():
        if v is None:
            continue
        parts = k.split(".")
        cur = nested
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return nested

def convert_row(row, casters):
    converted = {}
    for k, v in row.items():
        # Keep _id as a string
        if k == "_id":
            converted[k] = (v.strip() if v is not None else None)
            continue
            
        caster = casters.get(k)
        if caster:
            converted[k] = caster(v)
        else:
            # Default to string, empty string becomes None
            if v is None:
                converted[k] = None
            else:
                vv = v.strip()
                converted[k] = vv if vv != "" else None
                
    # Convert dotted keys to nested objects
    nested = nest_dotted_keys(converted)
    # Recursively remove None fields for cleanliness
    nested = drop_none_recursive(nested)
    return nested

def import_csv(coll, filepath, casters, batch_size=1000):
    count = 0
    ops = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = convert_row(row, casters)
            if doc:
                ops.append(InsertOne(doc))
                
            if len(ops) >= batch_size:
                res = coll.bulk_write(ops, ordered=False)
                count += res.inserted_count
                ops = []
                
        if ops:
            res = coll.bulk_write(ops, ordered=False)
            count += res.inserted_count
    return count
    
def main():
    parser = argparse.ArgumentParser(description="Import CSVs into MongoDB studentManagement.")
    parser.add_argument("--uri", default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--db", default="studentManagement", help="Database name")
    parser.add_argument("--data-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--drop", action="store_true", help="Drop collections before import")
    
    # This line MUST come before you use 'args'
    args = parser.parse_args()

    # Initialize the client inside the function
    client = MongoClient(args.uri)
    db = client[args.db]

    # Simple loop to process your directory
    if os.path.isdir(args.data_dir):
        for filename in os.listdir(args.data_dir):
            if filename.endswith(".csv"):
                print(f"Processing {filename}...")
                coll_name = filename.replace(".csv", "")
                coll = db[coll_name]
                
                if args.drop:
                    coll.drop()
                
                path = os.path.join(args.data_dir, filename)
                # Note: 'casters' is empty here; add mappings if you need specific types
                count = import_csv(coll, path, casters={}) 
                print(f"Successfully imported {count} documents into {coll_name}.")
    else:
        print(f"Error: Directory not found: {args.data_dir}")
    
    client.close()
 
if __name__ == "__main__":
    main()    