import os
import argparse
import numpy as np
import sys

# Handle MoviePy imports with fallbacks for different package structures
try:
    # Try to import from moviepy.editor (most common approach)
    from moviepy.editor import VideoFileClip, VideoClip, concatenate_videoclips, concatenate_audioclips, AudioFileClip
    print("Successfully imported MoviePy components")
except ImportError:
    try:
        # Try direct imports as a fallback
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.VideoClip import VideoClip
        from moviepy.video.compositing.concatenate import concatenate_videoclips
        from moviepy.audio.AudioClip import concatenate_audioclips
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        print("Successfully imported MoviePy components through direct imports")
    except ImportError:
        print("ERROR: Could not import MoviePy. Please install it with:")
        print("pip install moviepy")
        print("\nIf it's already installed, your installation might be corrupted.")
        print("Try reinstalling with: pip uninstall moviepy && pip install moviepy")
        sys.exit(1)
import yaml
import whisper
import tempfile
import json
import time

# Default configuration
DEFAULT_CONFIG = {
    # Silence detection parameters
    "silence_threshold": -40,  # in dB, lower = more sensitive
    "min_silence_duration": 0.1,  # minimum duration of silence to cut (seconds)
    "padding": 0.1,  # seconds to keep before and after speech

    # Speech detection parameters
    "whisper_model": "base",  # whisper model to use ("tiny", "base", "small", "medium", "large")
    "min_segment_duration": 0.2,  # Minimum duration for a segment to keep
    "max_segment_gap": 0.5,  # Maximum gap between words to be considered in the same segment

    # Consistency parameters
    "consistent_min_gap": 0.3,       # Minimum gap (seconds) between segments in normalized output
    "consistent_min_segment": 1.0,    # Minimum segment duration (seconds) for consistent processing
    "consistent_max_segment": 20.0,   # Maximum segment duration (seconds) before splitting
    
    # Mid-sentence pause detection
    "detect_mid_sentence_pauses": True,  # Whether to detect and trim pauses inside a sentence
    "min_mid_sentence_pause": 0.5,      # Minimum duration (seconds) for a mid-sentence pause to cut
    "max_mid_sentence_pause": 5.0,      # Maximum duration (seconds) considered a pause (not a scene break)
    "pause_energy_threshold": -35,      # Energy threshold for detecting pauses (dB)
    
    # Filler word parameters
    "remove_fillers": True,  # Whether to remove filler words
    "filler_words": ["um", "uh", "er", "ah", "hmm", "like", "you know", "basically", "actually", "literally"],
    "filler_word_padding": 0.05,  # seconds to keep before and after filler words
    
    # Audio-based filler detection (for catching fillers missed in transcription)
    "audio_filler_detection": True,  # Whether to detect fillers based on audio characteristics
    "audio_filler_threshold": -20,   # Energy threshold for potential fillers (dB)
    "audio_filler_min_duration": 0.1, # Min duration for audio-based filler (seconds)
    "audio_filler_max_duration": 0.5, # Max duration for audio-based filler (seconds)
    "filler_energy_ratio": 0.6,     # Ratio of energy compared to surrounding speech
    "filler_detection_sensitivity": 0.7, # Higher values = more aggressive detection (0-1)
    
    # Rephrasing detection parameters
    "detect_rephrasing": True,  # Whether to detect and remove rephrasing/repetitions
    "rephrasing_max_words": 5,   # Maximum number of words to look for in a rephrasing pattern
    "rephrasing_time_window": 3.0, # Time window in seconds to look for repeated phrases
    "rephrasing_similarity_threshold": 0.8, # How similar phrases need to be to be considered repetition (0-1)
    "detect_word_repetitions": True,  # Whether to detect sequences of repeated words (e.g., "is it is it is it")
    "word_repetition_threshold": 2,   # Minimum number of repetitions to detect (e.g., 2 would detect repeated pairs)
    
    # Video processing parameters
    "output_codec": "libx264",  # video codec
    "audio_codec": "aac",  # audio codec
    "output_preset": "medium",  # encoding preset (slower = better quality)
    "bitrate": None,  # Video bitrate (None uses original bitrate)
    "audio_bitrate": None,  # Audio bitrate (None uses original bitrate)
    "threads": 2,  # Number of threads to use for encoding
    "maintain_resolution": True,  # Maintain original video resolution
    
    # Temporary file settings
    "temp_audio_file": "temp_audio.wav",  # temporary audio file
    
    # Debug options
    "debug": False,  # Enable debug output
    "save_transcript": False  # Save transcription to file
}

def load_config(config_path=None):
    """
    Load configuration from a YAML file or use defaults
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                user_config = yaml.safe_load(file)
                if user_config:  # Check if the config file is not empty
                    config.update(user_config)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default configuration")
            
    return config

def save_default_config(path="config.yaml"):
    """
    Save the default configuration to a YAML file with comments
    
    Args:
        path (str): Path to save the configuration file
    """
    # Add comments to the configuration file
    comments = [
        "# Silence Trimmer Configuration File",
        "#",
        "# Silence detection parameters:",
        "#   silence_threshold: Audio level (in dB) below which is considered silence. Lower values are more sensitive.",
        "#   min_silence_duration: Minimum duration of silence (in seconds) required to be cut.",
        "#   padding: Amount of time (in seconds) to keep before and after detected speech.",
        "#",
        "# Speech detection parameters:",
        "#   whisper_model: Whisper model to use for speech detection (tiny, base, small, medium, large).",
        "#     - tiny: Fastest but less accurate",
        "#     - base: Good balance between speed and accuracy",
        "#     - small/medium/large: More accurate but slower",
        "#   min_segment_duration: Minimum duration (in seconds) for a speech segment to be kept.",
        "#   max_segment_gap: Maximum gap (in seconds) between words to be considered in the same segment.",
        "#",
        "# Consistency parameters:",
        "#   consistent_min_gap: Minimum gap (seconds) between segments in normalized output.",
        "#   consistent_min_segment: Minimum segment duration (seconds) for consistent processing.",
        "#   consistent_max_segment: Maximum segment duration (seconds) before splitting.",
        "#",
        "# Mid-sentence pause detection:",
        "#   detect_mid_sentence_pauses: Whether to detect and remove long pauses within a sentence.",
        "#   min_mid_sentence_pause: Minimum pause duration to be cut (seconds).",
        "#   max_mid_sentence_pause: Maximum pause duration considered within a sentence (seconds).",
        "#   pause_energy_threshold: Energy threshold for detecting pauses (dB).",
        "#",
        "# Filler word parameters:",
        "#   remove_fillers: Whether to remove filler words (um, uh, etc.)",
        "#   filler_words: List of filler words to remove",
        "#   filler_word_padding: Seconds to keep before and after filler words",
        "#",
        "# Audio-based filler detection:",
        "#   audio_filler_detection: Whether to detect fillers based on audio patterns",
        "#   audio_filler_threshold: Energy threshold for potential fillers (dB)",
        "#   audio_filler_min_duration: Minimum duration for audio-based filler (seconds)",
        "#   audio_filler_max_duration: Maximum duration for audio-based filler (seconds)",
        "#   filler_energy_ratio: Ratio of energy compared to surrounding speech",
        "#   filler_detection_sensitivity: Higher values = more aggressive detection (0-1)",
        "#",
        "# Rephrasing detection parameters:",
        "#   detect_rephrasing: Whether to detect and remove repeated phrases like 'I thought it, I thought it'",
        "#   rephrasing_max_words: Maximum number of words to look for in a rephrasing pattern",
        "#   rephrasing_time_window: Time window (seconds) to look for repeated phrases",
        "#   rephrasing_similarity_threshold: How similar phrases need to be (0-1)",
        "#",
        "# Video processing parameters:",
        "#   output_codec: Video codec for output (libx264 recommended for quality and compatibility).",
        "#   audio_codec: Audio codec for output (aac recommended for quality and compatibility).",
        "#   output_preset: Encoding preset (slower = better quality): ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.",
        "#   bitrate: Video bitrate (None uses original bitrate, e.g. '8000k' for 8Mbps).",
        "#   audio_bitrate: Audio bitrate (None uses original bitrate, e.g. '192k' for 192kbps).",
        "#   threads: Number of CPU threads to use for encoding.",
        "#   maintain_resolution: Ensure that the output video maintains the original aspect ratio and dimensions.",
        "#",
        "# Temporary file settings:",
        "#   temp_audio_file: Path for temporary audio file.",
        "#",
        "# Debug options:",
        "#   debug: Enable debug output.",
        "#   save_transcript: Save speech recognition transcript to file.",
        "#"
    ]

    # Write comments first, then the configuration
    with open(path, 'w') as file:
        for comment in comments:
            file.write(comment + "\n")
        yaml.dump(DEFAULT_CONFIG, file, default_flow_style=False)
    
    print(f"Default configuration saved to {path}")
    print("You can edit this file to customize the silence trimming behavior.")

def normalize_segments(segments, min_gap=0.5, min_segment_duration=0.2, words_with_times=None):
    """
    Normalize segments to ensure consistent processing between runs
    
    Args:
        segments (list): List of (start_time, end_time) tuples
        min_gap (float): Minimum gap between segments to keep them separate
        min_segment_duration (float): Minimum duration for a segment to keep
        words_with_times (list): Optional list of (word, start_time, end_time) tuples
        
    Returns:
        list: Normalized list of segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    # Merge overlapping or very close segments
    normalized = []
    current_start, current_end = sorted_segments[0]
    
    for start, end in sorted_segments[1:]:
        # If this segment starts after the current ends + min_gap,
        # start a new segment
        if start > current_end + min_gap:
            # Check if segment should be kept
            segment_duration = current_end - current_start
            if segment_duration >= min_segment_duration:
                normalized.append((current_start, current_end))
            else:
                # Segment is too short, but check if it contains important words (single letters, numbers)
                has_important_word = False
                if words_with_times:
                    for word, w_start, w_end in words_with_times:
                        clean_word = word.lower().strip(".,?!:;-\"'()[]{}").strip()
                        # Check if word is within the current segment time range
                        if (w_start >= current_start and w_end <= current_end and 
                           (len(clean_word) == 1 or clean_word.isdigit())):
                            has_important_word = True
                            break
                
                if has_important_word:
                    normalized.append((current_start, current_end))
            
            current_start, current_end = start, end
        else:
            # Extend the current segment
            current_end = max(current_end, end)
    
    # Add the last segment using the same logic
    segment_duration = current_end - current_start
    if segment_duration >= min_segment_duration:
        normalized.append((current_start, current_end))
    else:
        # Segment is too short, but check if it contains important words
        has_important_word = False
        if words_with_times:
            for word, w_start, w_end in words_with_times:
                clean_word = word.lower().strip(".,?!:;-\"'()[]{}").strip()
                # Check if word is within the current segment time range
                if (w_start >= current_start and w_end <= current_end and 
                   (len(clean_word) == 1 or clean_word.isdigit())):
                    has_important_word = True
                    break
        
        if has_important_word:
            normalized.append((current_start, current_end))
    
    return normalized

def apply_consistent_segmentation(speech_segments, config, words_with_times=None):
    """
    Apply consistent segmentation to ensure reproducible results
    
    Args:
        speech_segments (list): List of detected speech segments
        config (dict): Configuration dictionary
        words_with_times (list): List of (word, start_time, end_time) tuples
        
    Returns:
        list: Consistently processed segments
    """
    # Apply normalization with stricter parameters
    normalized = normalize_segments(
        speech_segments, 
        min_gap=config.get("consistent_min_gap", 0.3),
        min_segment_duration=config.get("min_segment_duration", 0.2),
        words_with_times=words_with_times
    )
    
    # Add consistent segmentation parameters to config
    consistent_min_segment = config.get("consistent_min_segment", 1.0)
    consistent_max_segment = config.get("consistent_max_segment", 20.0)
    
    # Split very long segments for more consistent processing
    final_segments = []
    for start, end in normalized:
        duration = end - start
        
        # If segment is too long, split it into roughly equal chunks
        # that are no longer than consistent_max_segment
        if duration > consistent_max_segment:
            # Calculate number of chunks needed
            num_chunks = max(2, int(duration / consistent_max_segment) + 1)
            chunk_size = duration / num_chunks
            
            # Create chunks
            for i in range(num_chunks):
                chunk_start = start + (i * chunk_size)
                chunk_end = min(start + ((i + 1) * chunk_size), end)
                final_segments.append((chunk_start, chunk_end))
        else:
            final_segments.append((start, end))
    
    # Sort by start time and return
    return sorted(final_segments, key=lambda x: x[0])

def is_filler_word(word, filler_words):
    """
    Check if a word is a filler word
    
    Args:
        word (str): Word to check
        filler_words (list): List of filler words
        
    Returns:
        bool: True if the word is a filler word, False otherwise
    """
    # Convert to lowercase and strip punctuation for comparison
    clean_word = word.lower().strip(".,?!:;-\"'()[]{}").strip()
    return clean_word in filler_words or clean_word + "," in filler_words or clean_word + "." in filler_words

def get_phrase_similarity(phrase1, phrase2):
    """
    Calculate similarity between two phrases
    
    Args:
        phrase1 (str): First phrase
        phrase2 (str): Second phrase
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = phrase1.lower().split()
    words2 = phrase2.lower().split()
    
    # If one is empty, return 0
    if not words1 or not words2:
        return 0.0
    
    # Count matching words (allowing for slight variations)
    matches = 0
    for w1 in words1:
        # Clean word of punctuation
        w1_clean = w1.strip(".,?!:;-\"'()[]{}").strip()
        if not w1_clean:
            continue
            
        for w2 in words2:
            # Clean word of punctuation
            w2_clean = w2.strip(".,?!:;-\"'()[]{}").strip()
            if not w2_clean:
                continue
                
            # Consider words matching if they're the same or one is a prefix of the other
            # This handles cases like "thought" and "thought-" where the speaker cuts off
            if w1_clean == w2_clean or (len(w1_clean) >= 3 and w2_clean.startswith(w1_clean)) or (len(w2_clean) >= 3 and w1_clean.startswith(w2_clean)):
                matches += 1
                break
    
    # Calculate Jaccard similarity (intersection over union)
    unique_words = len(set([w.strip(".,?!:;-\"'()[]{}").strip() for w in words1 + words2 if w.strip(".,?!:;-\"'()[]{}").strip()]))
    if unique_words == 0:
        return 0.0
        
    similarity = matches / unique_words
    return similarity

def detect_word_repetitions(words_with_times, config):
    """
    Detect sequences of repeated words (e.g., "is it is it is it")
    
    Args:
        words_with_times (list): List of (word, start_time, end_time) tuples
        config (dict): Configuration dictionary
        
    Returns:
        list: List of segments to remove as (start_time, end_time) tuples
    """
    if not config.get("detect_word_repetitions", True) or len(words_with_times) < 4:  # Need at least 4 words to find a repetition
        return []
    
    # Clean words for comparison (lowercase, remove punctuation)
    clean_words = [(w.lower().strip(".,?!:;-\"'()[]{}").strip(), start, end) for w, start, end in words_with_times if w.strip()]
    
    # Segments to remove
    segments_to_remove = []
    
    # Look for patterns of 1-2 words that repeat
    for pattern_length in [1, 2]:
        i = 0
        while i < len(clean_words) - (pattern_length * 2):  # Need at least two occurrences of the pattern
            # Get the current pattern
            current_pattern = [clean_words[i+j][0] for j in range(pattern_length)]
            
            # Count repetitions of this pattern
            repetition_count = 1
            next_position = i + pattern_length
            
            while next_position + pattern_length <= len(clean_words):
                next_pattern = [clean_words[next_position+j][0] for j in range(pattern_length)]
                
                # Check if patterns match
                if current_pattern == next_pattern:
                    repetition_count += 1
                    next_position += pattern_length
                else:
                    break
            
            # If we found enough repetitions, mark all but the last occurrence for removal
            if repetition_count >= config.get("word_repetition_threshold", 2):
                # Get the start time of the first occurrence
                pattern_start = clean_words[i][1]
                # Get the end time of the second-to-last occurrence
                pattern_end = clean_words[next_position - pattern_length][2]
                
                # Mark this segment for removal (keeping only the last occurrence)
                segments_to_remove.append((pattern_start, pattern_end))
                
                # Move past this pattern
                i = next_position
            else:
                # Move to the next word
                i += 1
        
    return segments_to_remove

def detect_rephrasing(segments_with_text, config):
    """
    Detect rephrasing/repetitions in speech segments
    
    Args:
        segments_with_text (list): List of (start_time, end_time, text) tuples
        config (dict): Configuration dictionary
        
    Returns:
        list: List of segments to remove as (start_time, end_time) tuples
    """
    if not config["detect_rephrasing"] or len(segments_with_text) < 2:
        return []
        
    # Sort segments by start time
    segments_with_text.sort(key=lambda x: x[0])
    
    # Find repetitions
    segments_to_remove = []
    
    for i in range(len(segments_with_text)):
        start1, end1, text1 = segments_with_text[i]
        
        # Look at subsequent segments within the time window
        for j in range(i + 1, len(segments_with_text)):
            start2, end2, text2 = segments_with_text[j]
            
            # Check if within time window
            if start2 > end1 + config["rephrasing_time_window"]:
                break
                
            # Check similarity
            similarity = get_phrase_similarity(text1, text2)
            
            if similarity >= config["rephrasing_similarity_threshold"]:
                # Found rephrasing - mark the earlier segment for removal
                segments_to_remove.append((start1, end1))
                break
    
    return segments_to_remove

def detect_mid_sentence_pauses(audio_path, speech_segments, config):
    """
    Detect long pauses within a sentence like "and so (2 second delay) clearly"
    
    Args:
        audio_path (str): Path to audio file
        speech_segments (list): List of detected speech segments (start_time, end_time)
        config (dict): Configuration dictionary
        
    Returns:
        list: List of segments to cut as (start_time, end_time) tuples
    """
    if not config.get("detect_mid_sentence_pauses", True):
        return []
        
    print("Detecting mid-sentence pauses...")
    
    try:
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)
        
        # Get audio parameters
        sample_rate = audio_clip.fps
        duration = audio_clip.duration
        
        # Sample the audio at intervals to analyze energy levels
        step = 0.01  # 10ms steps for analysis
        samples = np.arange(0, duration, step)
        
        # Calculate energy levels throughout the audio
        energies = []
        for t in samples:
            try:
                # Get frame at time t (returns array with values -1.0 to 1.0)
                frame = audio_clip.get_frame(t)
                
                # Convert to mono if stereo
                if len(frame.shape) > 1:
                    frame = frame.mean(axis=1)
                
                # Calculate energy in dB (avoid log(0) error)
                energy = 20 * np.log10(max(np.abs(frame).mean(), 1e-10))
                energies.append(energy)
            except Exception as e:
                print(f"Error processing audio at time {t}: {e}")
                energies.append(-100)  # Very low value in case of error
        
        # Convert to numpy array for easier processing
        energies = np.array(energies)
        times = np.array(samples)
        
        # Find pauses within segments that are classified as speech
        pause_segments = []
        
        min_pause_duration = config.get("min_mid_sentence_pause", 0.5)
        max_pause_duration = config.get("max_mid_sentence_pause", 5.0)
        pause_threshold = config.get("pause_energy_threshold", -35)
        
        # Get average speech energy to refine pause detection
        speech_energies = []
        for start, end in speech_segments:
            indices = (times >= start) & (times <= end)
            if any(indices):
                speech_energies.extend(energies[indices])
        
        if speech_energies:
            avg_speech_energy = np.mean(speech_energies)
            # Make threshold relative to speech energy if needed
            if pause_threshold > -100:
                pause_threshold = min(pause_threshold, avg_speech_energy - 15)
        
        # Check for long pauses within each speech segment
        for segment_start, segment_end in speech_segments:
            # Skip segments that are too short to contain meaningful pauses
            if segment_end - segment_start < min_pause_duration * 2:
                continue
                
            # Get energy values for this segment
            segment_indices = (times >= segment_start) & (times <= segment_end)
            segment_times = times[segment_indices]
            segment_energies = energies[segment_indices]
            
            if len(segment_times) < 5:  # Need enough samples
                continue
                
            # Find regions with energy below threshold (potential pauses)
            potential_pause = False
            pause_start = None
            
            for i in range(len(segment_times)):
                time = segment_times[i]
                energy = segment_energies[i]
                
                # Start of potential pause
                if not potential_pause and energy < pause_threshold:
                    potential_pause = True
                    pause_start = time
                
                # End of potential pause
                elif potential_pause and (energy >= pause_threshold or i == len(segment_times) - 1):
                    pause_end = time
                    pause_duration = pause_end - pause_start
                    
                    # Check if pause is long enough but not too long
                    if min_pause_duration <= pause_duration <= max_pause_duration:
                        # Ensure the pause isn't at the very start or end of the segment
                        if (pause_start > segment_start + 0.2 and 
                            pause_end < segment_end - 0.2):
                            pause_segments.append((pause_start, pause_end))
                    
                    potential_pause = False
        
        # Clean up
        audio_clip.close()
        
        if config.get("debug", False):
            print(f"Found {len(pause_segments)} mid-sentence pauses")
            for i, (start, end) in enumerate(pause_segments):
                print(f"  Mid-sentence pause {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
        return pause_segments
        
    except Exception as e:
        print(f"Error during mid-sentence pause detection: {e}")
        return []

def detect_audio_based_fillers(audio_path, speech_segments, config):
    """
    Detect filler words based on audio characteristics
    
    Args:
        audio_path (str): Path to audio file
        speech_segments (list): List of (start_time, end_time) tuples for detected speech
        config (dict): Configuration dictionary
        
    Returns:
        list: List of detected filler segments as (start_time, end_time) tuples
    """
    if not config.get("audio_filler_detection", True):
        return []
    
    print("Analyzing audio to detect filler sounds missed by transcription...")
    
    try:
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)
        
        # Get audio parameters
        sample_rate = audio_clip.fps
        duration = audio_clip.duration
        
        # Sample the audio at small intervals to analyze energy levels
        step = 0.01  # 10ms steps for analysis
        samples = np.arange(0, duration, step)
        
        # Calculate energy levels throughout the audio
        energies = []
        for t in samples:
            try:
                # Get frame at time t (returns array with values -1.0 to 1.0)
                frame = audio_clip.get_frame(t)
                
                # Convert to mono if stereo
                if len(frame.shape) > 1:
                    frame = frame.mean(axis=1)
                
                # Calculate energy in dB (avoid log(0) error)
                energy = 20 * np.log10(max(np.abs(frame).mean(), 1e-10))
                energies.append(energy)
            except Exception as e:
                print(f"Error processing audio at time {t}: {e}")
                energies.append(-100)  # Very low value in case of error
        
        # Convert to numpy array for easier processing
        energies = np.array(energies)
        times = np.array(samples)
        
        # Calculate average speech energy level from known speech segments
        speech_energies = []
        for start, end in speech_segments:
            indices = (times >= start) & (times <= end)
            if any(indices):
                speech_energies.extend(energies[indices])
        
        if not speech_energies:
            print("No speech segments with energy data found")
            return []
            
        avg_speech_energy = np.mean(speech_energies)
        speech_energy_std = np.std(speech_energies)
        
        # Adjust detection sensitivity based on config
        sensitivity = config.get("filler_detection_sensitivity", 0.7)
        
        # Define energy threshold based on config, avg speech energy, and sensitivity
        energy_threshold = config.get("audio_filler_threshold", -20)
        
        # Make threshold relative to speech energy
        adjusted_threshold = min(energy_threshold, 
                              avg_speech_energy * config.get("filler_energy_ratio", 0.6) - 
                              (sensitivity * speech_energy_std))
        
        # Detect potential filler sounds based on energy characteristics
        filler_segments = []
        
        # First, look for potential fillers within speech segments
        # This targets "um" and "uh" sounds that occur within a recognized speech segment
        for start, end in speech_segments:
            # Skip segments that are too short
            if end - start < config.get("audio_filler_min_duration", 0.1) * 2:
                continue
                
            # Get segment indices
            segment_indices = (times >= start) & (times <= end)
            segment_times = times[segment_indices]
            segment_energies = energies[segment_indices]
            
            if len(segment_times) < 5:  # Need enough samples
                continue
                
            # Find regions with characteristic filler energy profile
            # Fillers often have lower but consistent energy compared to regular speech
            potential_filler = False
            filler_start = None
            
            # Calculate local energy statistics to detect unusual patterns
            local_mean = np.mean(segment_energies)
            local_std = np.std(segment_energies)
            
            # Adjusted threshold for this specific segment
            local_threshold = min(adjusted_threshold, local_mean - (sensitivity * local_std))
            
            for i in range(len(segment_times)):
                time = segment_times[i]
                energy = segment_energies[i]
                
                # Start of potential filler - energy below threshold but above silence
                if not potential_filler and (
                    energy < local_mean - (sensitivity * local_std) and 
                    energy > config.get("silence_threshold", -40)
                ):
                    potential_filler = True
                    filler_start = time
                
                # End of potential filler - energy returns to normal range
                elif potential_filler and (
                    energy >= local_mean - (0.5 * local_std) or 
                    i == len(segment_times) - 1
                ):
                    filler_end = time
                    filler_duration = filler_end - filler_start
                    
                    # Check if duration is in the expected range for fillers
                    if (config.get("audio_filler_min_duration", 0.1) <= filler_duration <= 
                        config.get("audio_filler_max_duration", 0.5)):
                        filler_segments.append((filler_start, filler_end))
                    
                    potential_filler = False
        
        # Next, look for potential fillers between speech segments
        # This targets the classic "um" or "uh" pauses between sentences
        for i in range(len(speech_segments) - 1):
            curr_end = speech_segments[i][1]
            next_start = speech_segments[i+1][0]
            
            # Skip if gap is too small or too large
            gap_duration = next_start - curr_end
            if gap_duration < config.get("audio_filler_min_duration", 0.1) or \
               gap_duration > config.get("audio_filler_max_duration", 0.5) * 2:
                continue
                
            # Analyze the gap between speech segments
            gap_indices = (times >= curr_end) & (times <= next_start)
            gap_energies = energies[gap_indices]
            gap_times = times[gap_indices]
            
            if len(gap_energies) < 3:  # Need enough samples to analyze
                continue
            
            # Find regions with energy above silence but below normal speech
            # This targets the typical energy profile of filler words
            potential_filler = False
            filler_start = None
            
            for j in range(len(gap_energies)):
                energy = gap_energies[j]
                time = gap_times[j]
                
                # Detect start of potential filler
                if not potential_filler and energy > config.get("silence_threshold", -40) and energy < avg_speech_energy:
                    potential_filler = True
                    filler_start = time
                
                # Detect end of potential filler
                elif potential_filler and (energy <= config.get("silence_threshold", -40) or energy >= avg_speech_energy or j == len(gap_energies) - 1):
                    filler_end = time
                    filler_duration = filler_end - filler_start
                    
                    # Check if duration is in the expected range for fillers
                    if (config.get("audio_filler_min_duration", 0.1) <= filler_duration <= 
                        config.get("audio_filler_max_duration", 0.5)):
                        filler_segments.append((filler_start, filler_end))
                    
                    potential_filler = False
        
        # Clean up
        audio_clip.close()
        
        if config.get("debug", False):
            print(f"Audio analysis found {len(filler_segments)} potential filler sounds")
            for i, (start, end) in enumerate(filler_segments):
                print(f"  Audio filler {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
        return filler_segments
        
    except Exception as e:
        print(f"Error during audio-based filler detection: {e}")
        return []

def detect_speech_segments(audio_path, config):
    """
    Use Whisper to detect speech segments in the audio
    
    Args:
        audio_path (str): Path to audio file
        config (dict): Configuration dictionary
        
    Returns:
        list: List of (start_time, end_time) tuples for speech segments
    """
    print(f"Loading Whisper model ({config['whisper_model']})...")
    model = whisper.load_model(config["whisper_model"])
    
    print("Transcribing audio to detect speech segments...")
    start_time = time.time()
    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True
    )
    transcription_time = time.time() - start_time
    print(f"Transcription completed in {transcription_time:.1f} seconds")
    
    # Debug: save the full transcription result
    if config["save_transcript"]:
        with open("transcription.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Transcription saved to transcription.json")
    
    # Extract word timestamps and build text for each segment
    word_segments = []
    filler_segments = []
    segment_texts = []  # Will store (start, end, text) for rephrasing detection
    words_with_times = []  # Will store (word, start, end) for word repetition detection
    
    # First, process at the segment level to get phrases
    for segment in result["segments"]:
        segment_start = segment.get("start", 0)
        segment_end = segment.get("end", 0)
        segment_text = segment.get("text", "").strip()
        
        if segment_start < segment_end and segment_text:
            segment_texts.append((segment_start, segment_end, segment_text))
    
    # Next, process at the word level for detailed timing
    # And identify important words (numbers, single letters)
    important_word_segments = []  # This will store segments with important words
    
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            if "start" in word_info and "end" in word_info and "word" in word_info:
                start = word_info["start"]
                end = word_info["end"]
                word = word_info["word"]
                
                # Store all words with their timestamps
                words_with_times.append((word, start, end))
                
                # Check if this is an important word (number or single letter)
                clean_word = word.strip(".,?!:;-\"'()[]{}").strip().lower()
                if clean_word.isdigit() or (len(clean_word) == 1 and clean_word.isalpha()):
                    # Add extra padding to important words to ensure they're captured
                    padding = max(config["padding"], 0.2)
                    important_word_segments.append((
                        max(0, start - padding),
                        end + padding
                    ))
                    if config.get("debug", False):
                        print(f"Found important word: '{word}' at {start:.2f}s - {end:.2f}s")
                
                # Check if this is a filler word
                if config["remove_fillers"] and is_filler_word(word, config["filler_words"]):
                    padding = config["filler_word_padding"]
                    filler_segments.append((max(0, start - padding), end + padding, word))
                else:
                    word_segments.append((start, end))
    
    # Detect rephrasing/repetitions if enabled
    rephrasing_segments = []
    if config["detect_rephrasing"] and segment_texts:
        print("Detecting rephrasing patterns...")
        rephrasing_segments = detect_rephrasing(segment_texts, config)
        
        if config["debug"] and rephrasing_segments:
            print(f"Found {len(rephrasing_segments)} potential rephrasing segments")
            for start, end in rephrasing_segments:
                # Find text for this segment
                for s_start, s_end, text in segment_texts:
                    if abs(s_start - start) < 0.1 and abs(s_end - end) < 0.1:
                        print(f"  Rephrasing: '{text}' at {start:.2f}s - {end:.2f}s")
                        break
    
    # Detect word repetitions (like "is it is it is it")
    word_repetition_segments = []
    if config.get("detect_word_repetitions", True) and words_with_times:
        print("Detecting word repetition patterns...")
        word_repetition_segments = detect_word_repetitions(words_with_times, config)
        
        if config["debug"] and word_repetition_segments:
            print(f"Found {len(word_repetition_segments)} word repetition segments")
            for start, end in word_repetition_segments:
                # Find words in this range
                words_in_segment = []
                for word, w_start, w_end in words_with_times:
                    if w_start >= start and w_end <= end:
                        words_in_segment.append(word)
                
                pattern_text = " ".join(words_in_segment)
                print(f"  Word repetition: '{pattern_text}' at {start:.2f}s - {end:.2f}s")
    
    # Merge close segments based on max_segment_gap
    if not word_segments:
        print("Warning: No speech segments detected. Check audio quality.")
        return []
    
    # Combine regular word segments with important word segments
    all_word_segments = word_segments + important_word_segments
    all_word_segments.sort(key=lambda x: x[0])  # Sort by start time
    
    merged_segments = []
    current_start, current_end = all_word_segments[0]
    
    # Define a more aggressive max_gap for important words
    important_word_max_gap = config["max_segment_gap"] * 2  # More aggressive merging
    
    for start, end in all_word_segments[1:]:
        # Check if this segment or previous segment contains an important word
        contains_important_word = False
        for important_start, important_end in important_word_segments:
            # Check if current segment or next segment contains an important word
            if (important_start <= end and important_end >= start) or \
               (important_start <= current_end and important_end >= current_start):
                contains_important_word = True
                break
        
        # Use a more aggressive gap if there's an important word
        max_gap = important_word_max_gap if contains_important_word else config["max_segment_gap"]
        
        # If this segment starts after the current ends + max_gap,
        # start a new segment
        if start > current_end + max_gap:
            # Add padding to the current segment
            padded_start = max(0, current_start - config["padding"])
            padded_end = current_end + config["padding"]
            merged_segments.append((padded_start, padded_end))
            current_start, current_end = start, end
        else:
            # Extend the current segment
            current_end = max(current_end, end)
    
    # Add the last segment
    padded_start = max(0, current_start - config["padding"])
    padded_end = current_end + config["padding"]
    merged_segments.append((padded_start, padded_end))
    
    # Define final_segments variable here before first use
    final_segments = merged_segments
    
    # Detect mid-sentence pauses
    mid_sentence_pauses = []
    if config.get("detect_mid_sentence_pauses", True):
        mid_sentence_pauses = detect_mid_sentence_pauses(audio_path, merged_segments, config)
        if mid_sentence_pauses:
            print(f"Found {len(mid_sentence_pauses)} mid-sentence pauses")
    
    # Use audio analysis to detect filler sounds missed by transcription
    audio_filler_segments = []
    if config.get("audio_filler_detection", True):
        audio_filler_segments = detect_audio_based_fillers(audio_path, merged_segments, config)
        
        if audio_filler_segments:
            print(f"Found {len(audio_filler_segments)} audio-based filler segments")
            filler_segments.extend([(start, end, "audio_filler") for start, end in audio_filler_segments])
    
    # Remove mid-sentence pauses if enabled, but be careful around important words
    if config.get("detect_mid_sentence_pauses", True) and mid_sentence_pauses:
        print(f"Removing {len(mid_sentence_pauses)} mid-sentence pauses")
        
        # Break segments at mid-sentence pauses, but be careful with important words
        segments_without_pauses = []
        for start, end in final_segments:
            # Check all pauses that might affect this segment
            segment_pauses = []
            for p_start, p_end in mid_sentence_pauses:
                if p_start > start and p_end < end:
                    # Check if this pause overlaps with any important word
                    pause_overlaps_important = False
                    for important_start, important_end in important_word_segments:
                        if (important_start <= p_end and important_end >= p_start):
                            pause_overlaps_important = True
                            break
                    
                    if not pause_overlaps_important:
                        segment_pauses.append((p_start, p_end))
            
            if not segment_pauses:
                # No pauses in this segment
                segments_without_pauses.append((start, end))
                continue
            
            # Sort pauses by start time
            segment_pauses.sort()
            
            # Create sub-segments that avoid the pauses
            current = start
            for p_start, p_end in segment_pauses:
                if p_start > current:
                    segments_without_pauses.append((current, p_start))
                current = p_end
            
            # Add final segment after the last pause
            if current < end:
                segments_without_pauses.append((current, end))
                
        final_segments = segments_without_pauses
    
    # Remove filler words if enabled, but be careful with important words
    if config["remove_fillers"] and filler_segments:
        print(f"Found {len(filler_segments)} filler words to remove")
        
        if config["debug"]:
            for start, end, word in filler_segments:
                print(f"  Filler word: '{word}' at {start:.2f}s - {end:.2f}s")
        
        # Remove filler segments by breaking up speech segments
        segments_after_fillers = []
        for start, end in final_segments:
            # Find all fillers that overlap with this segment
            overlapping_fillers = []
            for f_start, f_end, _ in filler_segments:
                # Check if filler overlaps with an important word
                filler_overlaps_important = False
                for important_start, important_end in important_word_segments:
                    if (important_start <= f_end and important_end >= f_start):
                        filler_overlaps_important = True
                        break
                
                # Only consider fillers that don't overlap with important words
                if not filler_overlaps_important and (f_start < end and f_end > start):
                    overlapping_fillers.append((f_start, f_end))
            
            if not overlapping_fillers:
                # No fillers in this segment
                segments_after_fillers.append((start, end))
                continue
            
            # Sort fillers by start time
            overlapping_fillers.sort()
            
            # Create segments avoiding fillers
            current = start
            for f_start, f_end in overlapping_fillers:
                if f_start > current:
                    # Add segment before the filler
                    segments_after_fillers.append((current, f_start))
                current = max(current, f_end)
            
            # Add final segment after last filler if needed
            if current < end:
                segments_after_fillers.append((current, end))
                
        final_segments = segments_after_fillers
    
    # Remove rephrasing and word repetitions with special care for important words
    # (similar approach for both)
    if config["detect_rephrasing"] and rephrasing_segments:
        print(f"Removing {len(rephrasing_segments)} rephrasing segments")
        
        # Remove rephrasing segments with special care for important words
        segments_after_rephrasing = []
        for start, end in final_segments:
            # Check if this segment contains any important words
            segment_has_important = False
            for important_start, important_end in important_word_segments:
                if important_start >= start and important_end <= end:
                    segment_has_important = True
                    break
            
            # If segment has important words, preserve it
            if segment_has_important:
                segments_after_rephrasing.append((start, end))
                continue
            
            # Otherwise, check if it's a rephrasing segment
            is_rephrasing = False
            for r_start, r_end in rephrasing_segments:
                # If there's significant overlap (>50%), consider it a rephrasing
                overlap_start = max(start, r_start)
                overlap_end = min(end, r_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                segment_duration = end - start
                
                if overlap_duration > 0.5 * segment_duration:
                    is_rephrasing = True
                    break
            
            if not is_rephrasing:
                segments_after_rephrasing.append((start, end))
                
        final_segments = segments_after_rephrasing
    
    # Filter segments that are too short, but KEEP segments with important words
    min_duration = config["min_segment_duration"]
    filtered_segments = []
    
    for start, end in final_segments:
        segment_duration = end - start
        
        # Always keep segments containing important words, regardless of duration
        segment_has_important = False
        for important_start, important_end in important_word_segments:
            # Check if important word falls within this segment
            if important_start >= start and important_end <= end:
                segment_has_important = True
                break
        
        # Keep if it's long enough OR contains important words
        if segment_duration >= min_duration or segment_has_important:
            filtered_segments.append((start, end))
    
    if config["debug"]:
        print(f"Found {len(word_segments)} individual words")
        print(f"Found {len(important_word_segments)} important words (numbers/letters)")
        print(f"Merged into {len(merged_segments)} segments")
        print(f"After filtering: {len(filtered_segments)} segments")
    
    # Apply consistent segmentation, but protect important words
    final_segments = []
    for start, end in filtered_segments:
        # Check if this segment contains any important words
        segment_has_important = False
        for important_start, important_end in important_word_segments:
            if important_start >= start and important_end <= end:
                segment_has_important = True
                break
        
        # For segments with important words, keep them as is
        if segment_has_important:
            final_segments.append((start, end))
            continue
        
        # For other segments, apply normal consistent segmentation
        duration = end - start
        if duration > config.get("consistent_max_segment", 20.0):
            # Split very long segments
            num_chunks = max(2, int(duration / config.get("consistent_max_segment", 20.0)) + 1)
            chunk_size = duration / num_chunks
            
            for i in range(num_chunks):
                chunk_start = start + (i * chunk_size)
                chunk_end = min(start + ((i + 1) * chunk_size), end)
                final_segments.append((chunk_start, chunk_end))
        else:
            final_segments.append((start, end))
    
    # Sort by start time
    final_segments.sort(key=lambda x: x[0])
    
    print(f"Final number of segments: {len(final_segments)}")
    return final_segments

def trim_silence(input_video_path, output_path=None, config_path=None):
    """
    Trim silent parts from a video file
    
    Args:
        input_video_path (str): Path to input video file
        output_path (str): Path for output video (if None, will use input path + '_trimmed')
        config_path (str): Path to configuration file
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Load configuration
    config = load_config(config_path)
    
    if output_path is None:
        # Generate output path based on input path
        base, ext = os.path.splitext(input_video_path)
        output_path = f"{base}_trimmed{ext}"
    
    print(f"Processing video: {input_video_path}")
    
    try:
        # Load the video
        video = VideoFileClip(input_video_path)
        
        # Store original rotation and resolution
        original_rotation = getattr(video, 'rotation', 0)
        original_width, original_height = video.size
        
        print(f"Original dimensions: {original_width}x{original_height}")
        print(f"Original rotation: {original_rotation}")
        
        # Check if video has audio
        if not video.audio:
            print("Error: Video has no audio track. Cannot detect speech.")
            video.close()
            return False
        
        # Get video info
        print(f"Video properties:")
        print(f"  Duration: {video.duration:.2f} seconds")
        print(f"  Resolution: {video.size[0]}x{video.size[1]}")
        print(f"  FPS: {video.fps}")
        
        # Extract audio to a temporary file
        temp_audio = config["temp_audio_file"]
        print("Extracting audio from video...")
        video.audio.write_audiofile(temp_audio)
        
        # Detect speech segments
        speech_segments = detect_speech_segments(temp_audio, config)
        
        # Clean up the temporary audio file
        os.remove(temp_audio)
        
        if not speech_segments:
            print("No speech detected in the video. Outputting original video.")
            video.write_videofile(
                output_path, 
                codec=config["output_codec"],
                audio_codec=config["audio_codec"],
                preset=config["output_preset"],
                bitrate=config["bitrate"],
                audio_bitrate=config["audio_bitrate"],
                threads=config["threads"],
                ffmpeg_params=["-vf", f"scale={original_width}:{original_height}"] if config["maintain_resolution"] else None
            )
            video.close()
            return True
        
        # Create subclips for each speech segment
        print(f"Creating {len(speech_segments)} video segments...")
        subclips = []
        
        # Check if concatenate_videoclips can handle our video - test with small sample
        can_concatenate = True
        try:
            # Test if we can create a clip using the right method before proceeding
            if hasattr(video, 'subclip'):
                test_clip = video.subclip(0, min(1.0, video.duration))
            else:
                def get_test_frames(t):
                    return video.get_frame(t)
                test_clip = VideoClip(get_test_frames)
                test_clip.duration = min(1.0, video.duration)
                test_clip.fps = video.fps
                # Handle audio if present
                if video.audio is not None:
                    try:
                        # Create a new audio clip manually instead of using subclip
                        from moviepy.audio.AudioClip import AudioClip
                        
                        # Get the audio frames for the time range
                        def get_test_audio_frames(t):
                            return video.audio.get_frame(t)
                        
                        test_audio = AudioClip(get_test_audio_frames)
                        test_audio.duration = min(1.0, video.duration)
                        test_audio.fps = video.audio.fps
                        
                        test_clip.audio = test_audio
                    except Exception as audio_e:
                        print(f"Warning: Could not extract test audio: {audio_e}")
                        print("Test clip will not have audio.")
            
            # Clean up test clip
            test_clip.close()
        except Exception as e:
            print(f"Warning: Could not create test subclip: {e}")
            print("Falling back to manual frame extraction method")
            can_concatenate = False
            
        for i, (start, end) in enumerate(speech_segments):
            # Ensure we don't go beyond the video duration
            end = min(end, video.duration)
            if start >= end:
                continue
            
            print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            # Check if the video object has a subclip method (compatibility check)
            if hasattr(video, 'subclip'):
                subclip = video.subclip(start, end)
                subclips.append(subclip)
            else:
                # Alternative approach without using subclip method
                # Create a new clip that gets frames from the original video within the time range
                def get_frames_in_range(t):
                    return video.get_frame(start + t)
                
                # Create a new clip with the same settings as the original
                new_clip = VideoClip(get_frames_in_range)
                new_clip.duration = end - start
                new_clip.fps = video.fps
                
                # Ensure the clip keeps the original dimensions
                new_clip.size = video.size
                
                # Handle audio differently
                if video.audio is not None:
                    try:
                        # Try alternative audio extraction methods
                        # Method 1: Using set_start and set_end if available
                        try:
                            new_audio = video.audio.set_start(0)
                            new_audio = new_audio.set_end(end - start)
                            new_audio = new_audio.set_start(start)
                            new_clip.audio = new_audio
                        except AttributeError:
                            # Method 2: Create a new audio clip that samples from the original
                            from moviepy.audio.AudioClip import AudioClip
                            
                            # Function to get audio at time t from the original with offset
                            def get_audio_frames(t):
                                return video.audio.get_frame(start + t)
                            
                            new_audio = AudioClip(get_audio_frames)
                            new_audio.duration = end - start
                            new_audio.fps = video.audio.fps
                            
                            new_clip.audio = new_audio
                    except Exception as audio_e:
                        print(f"Warning: Could not extract audio for clip: {audio_e}")
                        print("This segment will not have audio.")
                
                subclips.append(new_clip)
        
        if not subclips:
            print("Error: No valid subclips created. Outputting original video.")
            video.write_videofile(
                output_path, 
                codec=config["output_codec"],
                audio_codec=config["audio_codec"],
                preset=config["output_preset"],
                bitrate=config["bitrate"],
                audio_bitrate=config["audio_bitrate"],
                threads=config["threads"],
                ffmpeg_params=["-vf", f"scale={original_width}:{original_height}"] if config["maintain_resolution"] else None
            )
            video.close()
            return True
        
        # Preserve original dimensions for all subclips
        for clip in subclips:
            clip.size = (original_width, original_height)
        
        # Concatenate all subclips or manually create final video
        if can_concatenate and len(subclips) > 0:
            try:
                print("Concatenating video segments...")
                final_video = concatenate_videoclips(subclips)
                # Ensure the final video has the correct dimensions
                final_video.size = (original_width, original_height)
            except Exception as e:
                print(f"Error concatenating clips: {e}")
                print("Falling back to manual frame extraction...")
                can_concatenate = False
        
        # Manual concatenation if moviepy concatenation failed
        if not can_concatenate:
            print("Creating final video using manual frame extraction...")
            
            # Calculate total duration
            total_duration = sum(clip.duration for clip in subclips)
            
            # Create a timeline mapping of when each clip should play
            timeline = []
            current_time = 0
            for clip in subclips:
                timeline.append((current_time, current_time + clip.duration, clip))
                current_time += clip.duration
            
            # Create function to get frame at time t
            def get_frame_at_time(t):
                for start, end, clip in timeline:
                    if start <= t < end:
                        return clip.get_frame(t - start)
                # Default frame (black) if outside any clip time
                # NOTE: This matches numpy's (height, width, channels) format
                return np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # Create new video clip
            final_video = VideoClip(get_frame_at_time)
            final_video.duration = total_duration
            final_video.fps = video.fps
            # Explicitly set the size to match the original
            final_video.size = (original_width, original_height)
            
            # Handle audio
            if video.audio is not None:
                audio_clips = [clip.audio for clip in subclips if clip.audio is not None]
                if audio_clips:
                    try:
                        final_video.audio = concatenate_audioclips(audio_clips)
                    except Exception as audio_e:
                        print(f"Warning: Could not concatenate audio clips: {audio_e}")
                        print("Final video will not have audio.")
        
        # Calculate statistics
        original_duration = video.duration
        trimmed_duration = final_video.duration
        saved_time = original_duration - trimmed_duration
        saved_percentage = (saved_time / original_duration) * 100
        
        # Write the final video
        print(f"Writing trimmed video to {output_path}...")
        
        # Use ffmpeg_params to force the correct resolution if needed
        ffmpeg_params = None
        if config["maintain_resolution"]:
            ffmpeg_params = ["-vf", f"scale={original_width}:{original_height}"]
            
            # If the original video had rotation metadata, preserve it
            if original_rotation != 0:
                ffmpeg_params.extend(["-metadata:s:v:0", f"rotate={original_rotation}"])
        
        final_video.write_videofile(
            output_path, 
            codec=config["output_codec"],
            audio_codec=config["audio_codec"],
            preset=config["output_preset"],
            bitrate=config["bitrate"],
            audio_bitrate=config["audio_bitrate"],
            threads=config["threads"],
            ffmpeg_params=ffmpeg_params
        )
        
        # Close all clips
        video.close()
        final_video.close()
        for clip in subclips:
            clip.close()
        
        print(f"Success! Trimmed video saved to: {output_path}")
        print(f"Original duration: {original_duration:.2f}s, Trimmed duration: {trimmed_duration:.2f}s")
        print(f"Removed {saved_time:.2f}s of silence ({saved_percentage:.1f}% reduction)")
        
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Trim silent parts from a video')
    parser.add_argument('video_file', nargs='?', help='Input video file')
    parser.add_argument('--output', '-o', help='Output video file')
    parser.add_argument('--config', '-c', help='Configuration file (YAML)')
    parser.add_argument('--create-config', action='store_true', 
                        help='Create a default configuration file and exit')
    parser.add_argument('--config-path', default='config.yaml',
                        help='Path to save the default configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--save-transcript', action='store_true',
                        help='Save speech recognition transcript to file')
    parser.add_argument('--remove-fillers', action='store_true',
                        help='Remove filler words like "um", "uh", etc.')
    parser.add_argument('--detect-rephrasing', action='store_true',
                        help='Detect and remove repeated phrases like "I thought it, I thought it"')
    parser.add_argument('--detect-word-repetitions', action='store_true',
                        help='Detect and remove repetitive words like "is it is it is it"')
    parser.add_argument('--detect-mid-sentence-pauses', action='store_true',
                        help='Detect and remove long pauses within sentences')
    parser.add_argument('--audio-filler-detection', action='store_true',
                        help='Use audio analysis to detect filler sounds (um, uh) missed in transcription')
    
    args = parser.parse_args()
    
    # Create default config file if requested
    if args.create_config:
        save_default_config(args.config_path)
        return
    
    # Check if a video file was provided
    if not args.video_file:
        parser.print_help()
        return
    
    # Check if the video file exists
    if not os.path.exists(args.video_file):
        print(f"Error: Video file '{args.video_file}' not found")
        return
    
    # Override debug settings if specified
    config = None
    if args.debug or args.save_transcript or args.remove_fillers or args.detect_rephrasing or args.detect_word_repetitions or args.detect_mid_sentence_pauses or args.audio_filler_detection:
        config = load_config(args.config)
        if args.debug:
            config["debug"] = True
        if args.save_transcript:
            config["save_transcript"] = True
        if args.remove_fillers:
            config["remove_fillers"] = True
        if args.detect_rephrasing:
            config["detect_rephrasing"] = True
        if args.detect_word_repetitions:
            config["detect_word_repetitions"] = True
        if args.detect_mid_sentence_pauses:
            config["detect_mid_sentence_pauses"] = True
        if args.audio_filler_detection:
            config["audio_filler_detection"] = True
        
        # Save to a temporary config file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
            yaml.dump(config, temp)
            temp_config_path = temp.name
        
        try:
            success = trim_silence(args.video_file, args.output, temp_config_path)
        finally:
            os.unlink(temp_config_path)
    else:
        success = trim_silence(args.video_file, args.output, args.config)
    
    if not success:
        print("Video processing failed")
        exit(1)

if __name__ == '__main__':
    main()