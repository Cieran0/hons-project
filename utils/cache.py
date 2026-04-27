import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime

class InferenceCache:
    """
    Cache inference results to avoid re-running inference.
    """
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, checkpoint_path, manifest_path, split, temperature=None):
        """Generate unique cache key based on inputs."""
        # Hash the checkpoint file to detect changes
        with open(checkpoint_path, 'rb') as f:
            checkpoint_hash = hashlib.md5(f.read()).hexdigest()
        # Hash the manifest
        with open(manifest_path, 'rb') as f:
            manifest_hash = hashlib.md5(f.read()).hexdigest()
        # Create key
        key_parts = [
            checkpoint_hash[:8],
            manifest_hash[:8],
            split,
            f"T{temperature:.2f}" if temperature is not None else "T1.00"
        ]
        return "_".join(key_parts)
    
    def save(self, probs, targets, image_ids, checkpoint_path, manifest_path, split, temperature=None):
        """Save inference results to cache."""
        cache_key = self._get_cache_key(checkpoint_path, manifest_path, split, temperature)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"
        # Save arrays
        np.savez_compressed(
            cache_file,
            probs=probs,
            targets=targets,
            image_ids=np.array(image_ids, dtype=object)
        )
        # Save metadata
        meta = {
            'cache_key': cache_key,
            'checkpoint_path': str(checkpoint_path),
            'manifest_path': str(manifest_path),
            'split': split,
            'temperature': temperature,
            'num_samples': len(probs),
            'created_at': datetime.now().isoformat(),
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f" Cached inference to {cache_file}")
        return cache_file
    
    def load(self, checkpoint_path, manifest_path, split, temperature=None):
        """Load inference results from cache if available."""
        cache_key = self._get_cache_key(checkpoint_path, manifest_path, split, temperature)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists() and meta_file.exists():
            # Load arrays
            data = np.load(cache_file, allow_pickle=True)
            probs = data['probs']
            targets = data['targets']
            image_ids = data['image_ids'].tolist()
            # Load metadata
            with open(meta_file) as f:
                meta = json.load(f)
            print(f" Loaded cached inference from {cache_file}")
            print(f"   Samples: {meta['num_samples']}, Created: {meta['created_at']}")
            return probs, targets, image_ids, meta
        else:
            print(f"No cache found for {cache_key}")
            return None, None, None, None
    
    def clear(self):
        """Clear all cached inference results."""
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        print(f"Cleared {self.cache_dir}")