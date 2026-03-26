import json
import random
import fire

def sample_json(input_path, sample_size, output_path, seed=None):
    """
    Sample a subset from a JSON file (list of dicts).

    Args:
        input_path (str): path to input JSON file
        sample_size (int): number of samples to draw
        output_path (str): path to save sampled JSON
        seed (int, optional): random seed for reproducibility
    """

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    # Handle case where sample_size > dataset size
    sample_size = min(sample_size, len(data))

    # Random sampling
    sampled_data = random.sample(data, sample_size)

    # Save sampled data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    print(f"Sampled {sample_size} records saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(sample_json)