import json
import os
def transform_prompt(original_prompt: str) -> str:
    """
    Extract Instruction + Input and flatten into a single prompt.
    """
    try:
        # Split sections
        if "### Instruction:" in original_prompt:
            instruction_part = original_prompt.split("### Instruction:")[1]
        else:
            return original_prompt.strip()

        if "### Input:" in instruction_part:
            instruction_text, rest = instruction_part.split("### Input:", 1)
        else:
            # No input case
            return instruction_part.strip()

        if "### Response:" in rest:
            input_text = rest.split("### Response:")[0]
        else:
            input_text = rest

        # Clean text
        instruction_text = instruction_text.strip().replace("\n", " ")
        input_text = input_text.strip().replace("\n", " ")

        # Merge (你給的格式是直接接在一起)
        new_prompt = instruction_text + input_text

        return new_prompt

    except Exception as e:
        print(f"Error processing prompt: {e}")
        return original_prompt.strip()


def convert_jsonl(input_jsonl_path, output_jsonl_path):
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)

            new_data = {
                "prompt": transform_prompt(data["prompt"]),
                "chosen": data["chosen"],
                "rejected": data["rejected"]
            }

            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for it in range(5):
        input_jsonl_path = f"/data2/chuanhsin0110/SPRec/models/SPRec/Goodreads_2048_0.00002/it{it}/data/nearest_ln_ches_scores/train.jsonl"
        output_jsonl_path = f"/data2/chuanhsin0110/CHES_DPO/experiments/data/Goodreads/it{it}/nearest_ln_ches_scores/train.jsonl"

        convert_jsonl(input_jsonl_path, output_jsonl_path)


    # convert_jsonl(input_jsonl_path, output_jsonl_path)
    