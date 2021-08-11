import torch
import torch.nn as nn

if __name__ == '__main__':
    from utils import get_centroids, get_cossim, calc_loss
elif __name__ == 'speaker_verification_ghostnet.ge2e_loss':
	from speaker_verification_ghostnet.utils import get_centroids, get_cossim, calc_loss


class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings):
        # Clamps all elements in input into the range [ min, max ]
        torch.clamp(self.w, min=1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss






