import torch
import torch.nn as nn
import torchvision.models as models
# https://discuss.pytorch.org/t/modify-resnet50-to-give-multiple-outputs/46905
# https://discuss.pytorch.org/t/what-is-the-correct-way-to-assign-new-value-to-buffer/104840

class ResNet(nn.Module):
    def __init__(self, ResNetSize, MIN_AGE, MAX_AGE, bins_inner_edges):
        super(ResNet, self).__init__()
        if ResNetSize == "ResNet34":
            self.model_resnet = models.resnet34()
        elif ResNetSize == "ResNet50":
            self.model_resnet = models.resnet50()
        elif ResNetSize == "ResNet101":
            self.model_resnet = models.resnet101()
        elif ResNetSize == "ResNet152":
            self.model_resnet = models.resnet152()
        num_features = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        # Output for age
        self.register_buffer('bins_inner_edges', bins_inner_edges)
        self.register_buffer('bins_means', self.get_mean_matrix(MIN_AGE, MAX_AGE))
        self.linears = nn.ModuleList([nn.Linear(num_features, bins_inner_edges.shape[1] + 1) for i in range(bins_inner_edges.shape[0])])

    def forward(self, x):
        x = self.model_resnet(x)

        bins_means = getattr(self, 'bins_means')
        M = bins_means.shape[0]
        L = bins_means.shape[1]
        # Age output
        age_preds_matrices = torch.zeros(size=(x.shape[0], M, L)).to(x.device)
        for linearIndex, m in enumerate(self.linears):
            pred = torch.softmax(m(x), dim=1)
            for imageIndex in range(x.shape[0]):
                age_preds_matrices[imageIndex, linearIndex] = pred[imageIndex]

        age_preds = torch.mean(torch.sum(age_preds_matrices * bins_means, dim=2), dim=1)

        return age_preds, age_preds_matrices

    def get_mean_matrix(self, MIN_AGE, MAX_AGE):
        bins_inner_edges = getattr(self, 'bins_inner_edges')
        M = bins_inner_edges.shape[0]
        L = bins_inner_edges.shape[1] + 1
        bins_means = torch.zeros(size=(M, L), dtype=torch.float32)

        for i in range(M):
            for j in range(1, L - 1):
                bins_means[i, j] = (bins_inner_edges[i, j] - 1 + bins_inner_edges[i, j-1]) / 2
            bins_means[i, 0] = (MIN_AGE + bins_inner_edges[i, 0] - 1) / 2
            bins_means[i, L-1] = (MAX_AGE + bins_inner_edges[i, L-2]) / 2
        return bins_means

