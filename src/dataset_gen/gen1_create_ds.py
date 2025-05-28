import os
import csv
import shutil
import re

def create_dataset_structure():
    """
    Creates a dataset structure from high confidence MIDI files.
    - Copies MIDI files to a 'midi' folder
    - Creates style text files with uncondensed style sequences
    - For AB style songs, only includes those with confidence > 0.98 for both A and B
    - Outputs a CSV with all rows that were added to the dataset
    """
    # Create necessary directories
    parent_dir = 'gen1-5k-real'
    midi_dir = os.path.join(parent_dir, 'midi')
    style_dir = os.path.join(parent_dir, 'style')
    
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    
    # Path to the high confidence CSV file
    csv_file = 'combined_results/high_confidence.csv'
    
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    
    # Output CSV file for tracking added rows
    output_csv = os.path.join(parent_dir, 'included_files.csv')
    included_rows = []
    
    # Read the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        for row in reader:
            file_id = row['file_id']
            prediction = row['prediction']
            significant_prediction = row.get('significant_prediction', '')
            
            # Check if this is an AB pattern and if so, verify confidence scores
            # Use significant_prediction for pattern detection as it's more reliable
            sections = re.findall(r'([A-Z])x\d+', significant_prediction if significant_prediction else prediction)
            pattern = ''.join(sections) if sections else ""
            
            # Skip AB patterns that don't meet confidence threshold
            if pattern == "AB":
                confidence_scores = row.get('confidence_scores', '')
                scores = re.findall(r'([A-Z]):(\d+\.\d+)', confidence_scores)
                score_dict = {style: float(score) for style, score in scores}
                
                # Skip if either A or B has confidence less than 0.98
                if 'A' not in score_dict or 'B' not in score_dict or score_dict['A'] < 0.98 or score_dict['B'] < 0.98:
                    continue
            
            # Create uncondensed style sequence
            style_sequence = expand_prediction(prediction)
            
            # Write style sequence to a text file
            style_file_path = os.path.join(style_dir, f"{file_id.replace('.mid', '')}.txt")
            with open(style_file_path, 'w') as style_file:
                style_file.write(style_sequence)
            
            # Copy MIDI file if it exists
            source_midi = f"combined_results/midi_files/{file_id}"  # Update this path to your MIDI files location
            target_midi = os.path.join(midi_dir, file_id)
            
            try:
                if os.path.exists(source_midi):
                    shutil.copy(source_midi, target_midi)
                    # Add this row to our included rows list
                    included_rows.append(row)
                else:
                    print(f"Warning: MIDI file {file_id} not found.")
            except Exception as e:
                print(f"Error copying {file_id}: {e}")
    
    # Write the included rows to a CSV file
    if included_rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(included_rows)
        print(f"Created CSV with {len(included_rows)} included files at {output_csv}")

def expand_prediction(prediction):
    """
    Expands a condensed prediction like "Ax10, Bx5" to an uncondensed sequence "AAAAAAAAAABBBBB"
    """
    # Remove any whitespace
    prediction = prediction.replace(" ", "")
    
    # Regular expression to match patterns like "Ax10" or "Bx5"
    pattern = r'([A-D])x(\d+)'
    
    expanded = ""
    for match in re.finditer(pattern, prediction):
        style = match.group(1)
        count = int(match.group(2))
        expanded += style * count
    
    return expanded

if __name__ == "__main__":
    create_dataset_structure()
    print("Dataset creation completed.")
