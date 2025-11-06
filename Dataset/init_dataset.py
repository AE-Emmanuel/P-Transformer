from datasets import load_dataset
import os

# Load the Python subset of the dataset, streaming it to avoid downloading everything.
dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", streaming=True, split="train")

# Define how many lines of code we want in our dataset
# This should result in a file of a few MBs, a good starting point.
num_lines_to_grab = 1000000 
output_file_path = "input.txt"
line_count = 0

print(f"Downloading a sample of the dataset to {output_file_path}...")

with open(output_file_path, "w", encoding="utf-8") as f:
    for example in dataset:
        # Each 'example' is a dictionary with a 'content' key holding the code
        code = example['content']
        # Write the code, adding a newline to separate it from the next file's content
        f.write(code + os.linesep) 
        line_count += code.count('\n')

        if line_count >= num_lines_to_grab:
            break

print(f"✅ Done! Created {output_file_path} with approximately {line_count} lines of code.")