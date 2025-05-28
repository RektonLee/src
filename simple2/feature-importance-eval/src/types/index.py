from dataclasses import dataclass
import torch
from torch_geometric.data import Data

@dataclass
class FeatureImportanceData:
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    temperature: torch.Tensor

    def to_torch_geometric_data(self) -> Data:
        return Data(
            x=self.x,
            pos=self.pos,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            temperature=self.temperature
        )