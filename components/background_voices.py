# components/background_voices.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List, Dict
import io
import librosa
import csv
from pathlib import Path

class BackgroundSoundDetector:
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self._load_yamnet_class_names()
    
    def _load_yamnet_class_names(self):
        """Load YAMNet class names from local CSV file"""
        class_map_path = Path(__file__).parent / "yamnet_class_map.csv"
        class_names = []
        
        with open(class_map_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                # CSV format: index,mid,display_name
                class_names.append(row[2])
        
        if len(class_names) != 521:
            raise ValueError(f"Expected 521 classes, got {len(class_names)}")
            
        return class_names

    def detect(self, audio_bytes: bytes) -> List[Dict]:
        # Load and preprocess audio
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        
        # Run YAMNet
        scores, _, _ = self.model(audio)
        scores = scores.numpy()
        
        results = []
        for i, window_scores in enumerate(scores):
            class_idx = np.argmax(window_scores)
            confidence = window_scores[class_idx]
            
            if confidence > 0.3:  # Confidence threshold
                results.append({
                    'label': self.class_names[class_idx],
                    'confidence': float(confidence),
                    'start_time': i * 0.975,  # Each window is 0.975s
                    'duration': 0.975
                })
        
        # Filter for emergency-relevant sounds (optional)
        emergency_keywords = [
            'siren', 'gunshot', 'explosion', 'scream', 'glass',
            'car horn', 'bark', 'crowd', 'baby cry', 'fire alarm',
            'crash', 'shout', 'alarm', 'emergency'
        ]
        
        filtered_results = [
            r for r in results
            if any(kw in r['label'].lower() for kw in emergency_keywords)
        ]
        
        return filtered_results

# Singleton instance
detector = BackgroundSoundDetector()

def detect_background_sounds(audio_bytes: bytes) -> List[Dict]:
    return detector.detect(audio_bytes)