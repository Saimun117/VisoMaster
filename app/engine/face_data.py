"""Qt-free target-face data container.

Lifted from `TargetFaceCardButton` (app/ui/widgets/widget_components.py:323),
stripped of all Qt/widget behaviour. Holds only the data the inference path
reads: `face_id`, recognition `embedding_store`, and the merged
`assigned_input_embedding` used to build the swap latent.
"""
from typing import Dict

import numpy as np


class TargetFace:
    def __init__(self, face_id: str, embedding_store: Dict[str, np.ndarray] | None = None):
        self.face_id = face_id
        # Key: recognition/embedding model name, Value: embedding of the TARGET face
        self.embedding_store: Dict[str, np.ndarray] = embedding_store or {}
        # Source faces assigned to this target. Key: input face id,
        # Value: {embedding model name: embedding}
        self.assigned_input_faces: Dict[str, Dict[str, np.ndarray]] = {}
        # Pre-merged saved embeddings, same value shape as assigned_input_faces
        self.assigned_merged_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        # Key: arcface model name, Value: merged SOURCE embedding to swap in
        self.assigned_input_embedding: Dict[str, np.ndarray] = {}

    def set_embedding(self, embedding_swap_model: str, embedding: np.ndarray):
        self.embedding_store[embedding_swap_model] = embedding

    def get_embedding(self, embedding_swap_model: str) -> np.ndarray:
        return self.embedding_store.get(embedding_swap_model, np.array([]))

    def calculate_assigned_input_embedding(self, control: dict):
        """Merge all assigned source embeddings per model into a single source
        embedding. `control['EmbMergeMethodSelection']` selects Mean/Median.

        Mirrors TargetFaceCardButton.calculate_assigned_input_embedding but
        takes `control` explicitly instead of reading it off main_window."""
        all_input_embeddings = []
        all_embedding_swap_models = set()

        for _, embedding_store in self.assigned_input_faces.items():
            if embedding_store:
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)

        for _, embedding_store in self.assigned_merged_embeddings.items():
            if embedding_store:
                all_embedding_swap_models.update(embedding_store.keys())
                all_input_embeddings.append(embedding_store)

        if len(all_input_embeddings) > 0:
            merge_method = control.get('EmbMergeMethodSelection', 'Mean')
            reducer = np.median if merge_method == 'Median' else np.mean
            self.assigned_input_embedding = {
                model: reducer(
                    [store[model] for store in all_input_embeddings if model in store],
                    axis=0,
                )
                for model in all_embedding_swap_models
            }
        else:
            self.assigned_input_embedding = {}
