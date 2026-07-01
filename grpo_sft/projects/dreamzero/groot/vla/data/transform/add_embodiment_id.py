from typing import Any, Dict
import torch
from groot.vla.data.transform.base import ModalityTransform
from groot.vla.data.schema import EmbodimentTag


class AddEmbodimentIDTransform(ModalityTransform):
    """
    Add embodiment_id to the data dict based on the embodiment_tag.
    """
    
    embodiment_tag_mapping: Dict[str, int]
    
    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Get embodiment tag from dataset metadata
        if self.dataset_metadata is not None and hasattr(self.dataset_metadata, 'embodiment_tag'):
            tag = self.dataset_metadata.embodiment_tag
            if isinstance(tag, EmbodimentTag):
                tag_value = tag.value
            else:
                tag_value = str(tag)
            
            # Map to embodiment_id
            if tag_value in self.embodiment_tag_mapping:
                data['embodiment_id'] = torch.tensor(self.embodiment_tag_mapping[tag_value], dtype=torch.long)
            else:
                raise ValueError(f"Embodiment tag '{tag_value}' not found in mapping: {self.embodiment_tag_mapping.keys()}")
        else:
            raise ValueError("Dataset metadata or embodiment_tag not set")
        
        return data
    
    def __call__(self, data: dict) -> dict:
        return self.apply(data)
