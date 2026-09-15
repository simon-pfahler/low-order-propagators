import torch
from qcd_ml.base.paths import PathBuffer
from qcd_ml.nn.pt import v_PT
from utility import generators


class Model_HL(torch.nn.Module):
    def __init__(self, nlayers, paths):
        super().__init__()
        self.paths = paths
        self.nlayers = nlayers
        self.npaths = len(self.paths)

        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(1, self.npaths, 4, 4, dtype=torch.cdouble)
                ),
                *[
                    torch.nn.Parameter(
                        torch.randn(
                            self.npaths, self.npaths, 4, 4, dtype=torch.cdouble
                        )
                    )
                    for _ in range(self.nlayers - 1)
                ],
                torch.nn.Parameter(
                    torch.randn(self.npaths, 1, 4, 4, dtype=torch.cdouble)
                ),
            ]
        )

    def forward(self, v, U):
        pt = v_PT(self.paths, U)

        # First dense and PT layers
        res = pt(torch.einsum("iost,...tc->o...sc", self.weights[0], v))
        # Residual connection
        res[0] += v

        for i in range(1, self.nlayers):
            # Intermediate layers with residual connections
            h = res[0].clone()
            res = pt(torch.einsum("iost,i...tc->o...sc", self.weights[i], res))
            res[0] += h

        # Final dense layer with residual connection
        res = torch.einsum("iost,i...tc->...sc", self.weights[-1], res) + res[0]

        return res


class Model_HC(torch.nn.Module):
    def __init__(self, nlayers, paths):
        super().__init__()
        self.paths = paths
        self.nlayers = nlayers
        self.npaths = len(self.paths)

        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(1, self.npaths, 16, dtype=torch.cdouble)
                ),
                *[
                    torch.nn.Parameter(
                        torch.randn(
                            self.npaths, self.npaths, 16, dtype=torch.cdouble
                        )
                    )
                    for _ in range(self.nlayers - 1)
                ],
                torch.nn.Parameter(
                    torch.randn(self.npaths, 1, 16, dtype=torch.cdouble)
                ),
            ]
        )

    def forward(self, v, U):
        pt = v_PT(self.paths, U)

        # First dense and PT layers
        res = pt(
            torch.einsum(
                "nst,ion,...tc->o...sc",
                generators,
                self.weights[0],
                v,
            )
        )
        # Residual connection
        res[0] += v

        for i in range(1, self.nlayers):
            # Intermediate layers with residual connections
            h = res[0].clone()
            res = pt(
                torch.einsum(
                    "nst,ion,i...tc->o...sc", generators, self.weights[i], res
                )
            )
            res[0] += h

        # Final dense layer with residual connection
        res = (
            torch.einsum(
                "nst,ion,i...tc->...sc", generators, self.weights[-1], res
            )
            + res[0]
        )

        return res


class Model_restricted(torch.nn.Module):
    def __init__(self, nlayers, paths):
        super().__init__()
        self.paths = paths
        self.nlayers = nlayers
        self.npaths = len(self.paths)

        self.overall_factor = torch.nn.Parameter(
            torch.ones(1, dtype=torch.cdouble)
        )

        self.weights = torch.nn.Parameter(
            torch.randn(self.nlayers, 2, dtype=torch.cdouble)
        )

    def forward(self, v, U):
        pt = [PathBuffer(U, pi) for pi in self.paths]

        for i in range(self.nlayers):
            v_pts = [pti.v_transport(v) for pti in pt]

            curr_terms = torch.zeros_like(v)
            for mu in range(4):
                w_plus = (
                    self.weights[i, 0] * generators[0]
                    + self.weights[i, 1] * generators[mu + 1]
                )
                w_minus = (
                    self.weights[i, 0] * generators[0]
                    - self.weights[i, 1] * generators[mu + 1]
                )
                curr_terms += torch.einsum(
                    "ij,...jc->...ic", w_plus, v_pts[2 * mu + 1]
                )
                curr_terms += torch.einsum(
                    "ij,...jc->...ic", w_minus, v_pts[2 * mu + 2]
                )
            v = v + curr_terms

        return self.overall_factor * v
