# Music Generation Debug Analysis

## Issues Identified

### 1. **Insufficient Prompt Length**
- **Current**: Only using 32 tokens as decoder prompt
- **Problem**: This is too short to capture the full musical structure, especially for pieces with multiple sections
- **Solution**: Increase prompt length to at least 64-128 tokens to include more structural context

### 2. **Missing Structural Guidance in Generation**
- **Current**: The decoder prompt doesn't preserve section markers (A_SECTION_TOKEN, B_SECTION_TOKEN, etc.)
- **Problem**: The model loses track of which section it should be generating
- **Solution**: Extract section tokens from encoder context and include them in the decoder prompt

### 3. **Context-Prompt Mismatch**
- **Current**: Encoder context contains structured snippets with section tokens, but decoder prompt is just raw music tokens
- **Problem**: The cross-attention mechanism can't properly align structure with generation
- **Solution**: Build decoder prompts that mirror the structure found in encoder context

### 4. **No Structure Enforcement During Generation**
- **Current**: Generation proceeds without checking if it's following the intended form
- **Problem**: Model can get stuck repeating the first section pattern
- **Solution**: Implement guided generation that injects section tokens at appropriate points

### 5. **Token Combination Issues**
- **Current**: `combine_sequences_with_timing` removes piano prefix tokens but they're added back inconsistently
- **Problem**: This can cause tokenization misalignment
- **Solution**: Consistent handling of special tokens throughout the pipeline

## Recommended Fixes

### Quick Fixes (Minimal Changes)
1. Increase prompt length from 32 to 64+ tokens
2. Ensure section tokens from encoder are preserved in decoder prompt
3. Add logging to track which sections are being generated

### Medium Fixes (Moderate Changes)
1. Implement structure-aware prompt building:
   ```python
   # Extract structure from encoder context
   # Build decoder prompt that includes relevant section markers
   # Ensure first 10-20 tokens include structural information
   ```

2. Add generation monitoring:
   ```python
   # Track which section tokens appear in generated output
   # Log when generation switches between sections
   # Alert if generation gets stuck in one section
   ```

### Advanced Fixes (Significant Changes)
1. **Guided Generation**: Implement a generation strategy that:
   - Tracks the intended musical form (e.g., ABA, ABAB)
   - Injects section tokens at appropriate intervals
   - Uses the encoder context to determine when to transition

2. **Structure-Aware Sampling**: Modify the sampling process to:
   - Increase probability of section tokens at structural boundaries
   - Decrease probability of repeating the same section too many times
   - Use the encoder's structure as a template

3. **Prompt Engineering**: Create better prompts by:
   - Including a "structure summary" at the beginning
   - Using special tokens to indicate transitions
   - Preserving timing information from the encoder

## Debugging Steps

1. **Verify Tokenization**:
   - Print decoder tokens to ensure section markers are present
   - Check if encoder context properly encodes structure
   - Verify that special tokens are correctly mapped

2. **Monitor Generation**:
   - Log every 50 tokens during generation to see patterns
   - Track section token appearances
   - Measure how long each section lasts

3. **Test Different Configurations**:
   - Try different prompt lengths (32, 64, 128, 256)
   - Test with/without section tokens in prompt
   - Experiment with different temperatures

## Example Implementation

```python
def build_structured_prompt(encoder_ids, decoder_ids, tokenizer):
    """Build a decoder prompt that preserves structural information."""
    
    # Find section tokens in encoder
    section_tokens = extract_section_tokens(encoder_ids, tokenizer)
    
    # Get musical content from decoder
    music_tokens = decoder_ids[:64]  # Use more tokens
    
    # Interleave section markers with music
    structured_prompt = []
    for section in section_tokens:
        structured_prompt.append(section)
        # Add some music tokens after each section marker
        structured_prompt.extend(music_tokens[:16])
        music_tokens = music_tokens[16:]
    
    return structured_prompt
```

## Next Steps

1. Start with increasing prompt length and adding logging
2. Implement structure extraction from encoder context
3. Test generation with structured prompts
4. If issues persist, implement guided generation with section token injection