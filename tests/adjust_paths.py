import os
import re
from collections import Counter

def count_files_per_style(directory):
    """
    Count the number of files per music style, where the style is the prefix
    before the first underscore in the filename (e.g., 'A' in 'A_00001.mid').
    """
    style_counter = Counter()
    for fname in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, fname)):
            match = re.match(r"([A-Z]+)_", fname)
            if match:
                style = match.group(1)
                style_counter[style] += 1
    for style, count in sorted(style_counter.items()):
        print(f"Style '{style}': {count} files")

# Example usage:
count_files_per_style("datasets/synth_midi_90k/style")
