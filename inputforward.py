from pymonntorch import Behavior
import torch

# _____________________________________________________Full connectivity_____________________________________________________________________


class FullConnectivityFirstOption(Behavior):
    def initialize(self, synapse):
        self.J0 = self.parameter("J0", None)
        self.alpha = self.parameter("alpha", 100) / 100

        self.N = synapse.src.size

        synapse.W = synapse.matrix(self.J0 / self.N)
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


class FullConnectivitySecondOption(Behavior):
    def initialize(self, synapse):
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100
        self.J0 = self.parameter("J0", 1)
        self.alpha = self.parameter("alpha", 100) / 100

        self.N = synapse.src.size

        mean = self.J0 / self.N
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


# _______________________________________Random connectivity: fixed coupling probability________________________________________________


class Scaling(Behavior):
    def initialize(self, synapse):
        self.p = self.parameter("p", None)
        self.J0 = self.parameter("J0", None)
        self.alpha = self.parameter("alpha", 100) / 100
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100

        self.N = synapse.src.size
        self.C = self.p * self.N

        mean = self.J0 / self.C
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        self.test = torch.rand_like(synapse.W)
        synapse.W[self.test > (self.p)] = 0
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


# __________________________________Random connectivity: fixed number of presynaptic partners________________________________________________


class FixedAAndFinite(Behavior):
    def initialize(self, synapse):
        self.J0 = self.parameter("J0", None)
        self.C = self.parameter("C", None)
        self.alpha = self.parameter("alpha", 100) / 100
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100

        self.total_size_of_matrix = synapse.src.size * synapse.dst.size
        mean = self.J0 / self.C
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        random_indices = torch.zeros_like(synapse.W)
        for i in range(synapse.dst.size):
            random_indices[:, i] = torch.randperm(synapse.src.size)
        synapse.W[random_indices > self.C] = 0
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha
