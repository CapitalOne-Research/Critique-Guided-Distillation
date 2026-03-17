import json
import os
import random

def load_data(path: str, limit: int = None) -> list:
    """Loads data from a JSON or JSONL file."""
    # ... (this function is unchanged)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    
    try:
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
        
    return data[:limit] if limit else data

def separate_samples_by_conclusion(data: list) -> tuple[list, list]:
    """Separates data into two lists based on the conclusion."""
    # ... (this function is unchanged)
    right_samples = []
    wrong_samples = []

    for ex in data:
        input_text = ex.get("input", "").lower().strip()
        
        if input_text.endswith("conclusion: right"):
        # if input_text.endswith("conclusion: right [end]"):
            right_samples.append(ex)
        elif input_text.endswith("conclusion: wrong"):
        # elif input_text.endswith("conclusion: wrong [end]"):
            wrong_samples.append(ex)
            
    print(f"Found {len(right_samples)} 'right' samples and {len(wrong_samples)} 'wrong' samples.")
    return right_samples, wrong_samples

def save_json_dataset(data: list, base_path: str, ratio_name: str):
    """Saves a dataset to a JSON file with a descriptive name."""
    # ... (this function is unchanged)
    if not data:
        print(f"Skipping {ratio_name} dataset as it would be empty.")
        return
        
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(directory, f"{name}_{ratio_name}{ext}")
    
    random.shuffle(data)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4)
    print(f"✅ Saved {len(data)} samples for ratio '{ratio_name}' to {output_path}")

def create_and_save_controlled_ratios(right_samples: list, wrong_samples: list, base_output_path: str):
    """
    Creates and saves JSON files with different ratios, ensuring all files have
    the same total number of samples, limited by the smallest category.
    """
    min_count = min(len(right_samples), len(wrong_samples))
    if min_count == 0:
        print("❌ Cannot create datasets because one of the categories has zero samples.")
        return
        
    print(f"🔬 All datasets will be controlled to a size of {min_count} samples.")

    # 1. 100% 'right' samples
    data_100_right = random.sample(right_samples, min_count)
    save_json_dataset(data_100_right, base_output_path, "100_right_controlled")

    # 2. 100% 'wrong' samples
    # No sampling needed as this is the smallest group
    data_100_wrong = wrong_samples
    save_json_dataset(data_100_wrong, base_output_path, "100_wrong_controlled")

    # 3. 50% 'right' - 50% 'wrong'
    half_count = min_count // 2
    right_part = random.sample(right_samples, half_count)
    wrong_part = random.sample(wrong_samples, half_count)
    # Ensure exact min_count even if it's odd
    if min_count % 2 != 0:
        right_part.append(random.choice(right_samples))
    data_50_50 = right_part + wrong_part
    save_json_dataset(data_50_50, base_output_path, "50_right_50_wrong_controlled")

    # 4. 25% 'right' - 75% 'wrong'
    num_right_25 = round(min_count * 0.25)
    num_wrong_75 = min_count - num_right_25
    data_25_75 = random.sample(right_samples, num_right_25) + random.sample(wrong_samples, num_wrong_75)
    save_json_dataset(data_25_75, base_output_path, "25_right_75_wrong_controlled")

    # 5. 75% 'right' - 25% 'wrong'
    num_right_75 = round(min_count * 0.75)
    num_wrong_25 = min_count - num_right_75
    data_75_25 = random.sample(right_samples, num_right_75) + random.sample(wrong_samples, num_wrong_25)
    save_json_dataset(data_75_25, base_output_path, "75_right_25_wrong_controlled")


# --- Main Execution ---
if __name__ == "__main__":
    data_path = "webinstruct_filter/100_300K/03_refined_dataset.json"
    output_base_path = "webinstruct_filter/100_300K/03_refined_dataset_filtered.json"

    raw_data = load_data(data_path)
    if raw_data:
        right_samples, wrong_samples = separate_samples_by_conclusion(raw_data)
        # Call the new controlled ratio function
        create_and_save_controlled_ratios(right_samples, wrong_samples, output_base_path)
    else:
        print("No data loaded. Exiting.")