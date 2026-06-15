import numpy as np

from .base import ProteinEmbedder

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class DummyEmbedder(ProteinEmbedder):

    def _embed_sequence(self, seq):
        sequence = str(seq).upper()
        aa_counts = np.zeros(len(AMINO_ACIDS))
        for i, aa in enumerate(AMINO_ACIDS):
            aa_counts[i] = sequence.count(aa)
        if len(sequence) > 0:
            aa_counts /= len(sequence)
        return aa_counts
