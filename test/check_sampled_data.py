import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_record(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Convert a record into a hashable normalized tuple.
    Strips leading/trailing whitespace for safer duplicate detection.
    """
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    return (instruction, input_text, output)


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of records.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Record at index {i} is not a JSON object.")

    return data


def count_duplicates(records: List[Dict[str, Any]]) -> None:
    total = len(records)

    # Exact duplicate records
    normalized_records = [normalize_record(r) for r in records]
    record_counter = Counter(normalized_records)

    duplicated_records = {k: v for k, v in record_counter.items() if v > 1}
    total_duplicate_groups = len(duplicated_records)

    # 幾筆資料是重複的：所有 count > 1 的筆數總和
    duplicated_entry_count = sum(count for count in record_counter.values() if count > 1)

    # 幾筆資料沒重複：只出現 1 次的筆數總和
    unique_entry_count = sum(count for count in record_counter.values() if count == 1)

    print("=" * 80)
    print(f"Total records: {total}")
    print("=" * 80)

    print("\n[Exact duplicate records]")
    print(f"Number of duplicate groups: {total_duplicate_groups}")

    if duplicated_records:
        for i, (record, count) in enumerate(duplicated_records.items(), start=1):
            instruction, input_text, output = record
            print(f"\nDuplicate group #{i} | repeated {count} times")
            print(f"instruction: {instruction}")
            print(f"input      : {input_text}")
            print(f"output     : {output}")
    else:
        print("No exact duplicate records found.")

    print("\n" + "=" * 80)
    print("[Summary]")
    print(f"Duplicated entries: {duplicated_entry_count}")
    print(f"Non-duplicated entries: {unique_entry_count}")


def main():
    file_path = input("Please enter the JSON file path: ").strip()
    records = load_json_file(file_path)
    print(f"Loaded {len(records)} records from {file_path}")
    # count_duplicates(records)


if __name__ == "__main__":
    main()