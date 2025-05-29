import csv
import re
from collections import Counter

def analyze_music_styles():
    """
    Analyzes the distribution of music styles from the high_confidence.csv file.
    Extracts patterns like AB, ABA, etc. from the significant_prediction column.
    Also calculates average confidence scores for each style class.
    Identifies songs with AB pattern that have high confidence (>0.95) for both A and B.
    """
    style_patterns = []
    # Dictionary to store confidence scores for each pattern and style class
    pattern_confidences = {}
    # Track AB songs with high confidence in both styles
    ab_high_confidence_count = 0
    total_ab_count = 0
    
    try:
        with open('combined_results/high_confidence.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract the significant prediction
                sig_pred = row['significant_prediction']
                
                # Use regex to find all sections (like Ax440, Bx1317)
                sections = re.findall(r'([A-Z])x\d+', sig_pred)
                
                if sections:
                    # Create a pattern like 'AB', 'ABA', etc.
                    pattern = ''.join(sections)
                    style_patterns.append(pattern)
                    
                    # Extract confidence scores
                    confidence_scores = row.get('confidence_scores', '')
                    if confidence_scores:
                        # Initialize pattern in dictionary if not exists
                        if pattern not in pattern_confidences:
                            pattern_confidences[pattern] = {}
                        
                        # Parse confidence scores like "A:0.983, B:0.946"
                        scores = re.findall(r'([A-Z]):(\d+\.\d+)', confidence_scores)
                        score_dict = {style: float(score) for style, score in scores}
                        
                        # Check if this is an AB pattern with high confidence in both A and B
                        if pattern == 'AB':
                            total_ab_count += 1
                            if 'A' in score_dict and 'B' in score_dict and score_dict['A'] > 0.98 and score_dict['B'] > 0.98:
                                ab_high_confidence_count += 1
                        
                        # Store all scores for averaging later
                        for style, score in scores:
                            if style not in pattern_confidences[pattern]:
                                pattern_confidences[pattern][style] = []
                            pattern_confidences[pattern][style].append(float(score))
    
        # Count the occurrences of each pattern
        pattern_counts = Counter(style_patterns)
        
        print("Music Style Distribution:")
        for pattern, count in pattern_counts.items():
            print(f"{pattern}: {count} pieces ({count/len(style_patterns)*100:.2f}%)")
            
            # Print average confidence scores for this pattern
            if pattern in pattern_confidences:
                print("  Average confidence scores:")
                for style, scores in pattern_confidences[pattern].items():
                    avg_score = sum(scores) / len(scores)
                    print(f"    Style {style}: {avg_score:.3f}")
                
                # Print special statistics for AB pattern
                if pattern == 'AB':
                    print(f"  AB songs with confidence > 0.95 for both A and B: {ab_high_confidence_count} out of {total_ab_count} ({ab_high_confidence_count/total_ab_count*100:.2f}%)")
        
        print(f"\nTotal analyzed pieces: {len(style_patterns)}")
        
    except FileNotFoundError:
        print("Error: combined_results/high_confidence.csv file not found.")
    except Exception as e:
        print(f"Error analyzing music styles: {e}")

if __name__ == "__main__":
    analyze_music_styles()
