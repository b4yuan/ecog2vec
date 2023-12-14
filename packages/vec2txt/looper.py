# run_train.py
import subprocess

# Number of times to run train.py
num_runs = 30
wer_values = []

for _ in range(num_runs):
    # Run train.py using subprocess and capture the output
    result = subprocess.run(["python3", "train_on_preprocessed_hg.py"], capture_output=True, text=True)
    
    # Capture each line of the output
    output_lines = result.stdout.splitlines()

    # Extract the wer value from the output
    wer_value = None
    for line in output_lines:
        if line.startswith("Avg test WER:"):
            wer_value = float(line.split(":")[-1].strip())
            break

    # Check if the wer value was found
    if wer_value is not None:
        wer_values.append(wer_value)
    else:
        print(f"Warning: test WER not found in the output of run {_ + 1}")

# Print or save the wer_values as needed
print("WER values:", wer_values)

# Save the values to a file (optional)
with open("wer_values.txt", "w") as file:
    for value in wer_values:
        file.write(f"{value}\n")
