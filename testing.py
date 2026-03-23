import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


def normalize(x, bounds):
    return (x - bounds[0]) / (bounds[1] - bounds[0])

def unnormalize(x, bounds):
    return x * (bounds[1] - bounds[0]) + bounds[0]

def f(x):
    return torch.sin(3 * x) + x**2

train_x = torch.rand(5, 1) * 4 - 2
train_y = f(train_x)

model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

def plot_iteration(model, train_x, train_y, f, bounds, iteration):
    model.eval()

    x = torch.linspace(bounds[0].item(), bounds[0, 0].item(), 200).unsqueeze(-1)

    y_true = f(x)

    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

    plt.figure(figsize=(8, 5))

    plt.plot(x.numpy(), y_true.numpy(), label="True function", linestyle="dashed")

    plt.plot(x.numpy(), mean.numpy(), label="GP mean")

    plt.fill_between(
        x.squeeze().numpy(),
        (mean - 2 * std).squeeze().numpy(),
        (mean + 2 * std).squeeze().numpy(),
        alpha=0.3,
        label="Uncertainty"
    )

    plt.scatter(train_x.numpy(), train_y.numpy(), color="red", label="Observations")

    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.show()

for i in range(10):
    EI = LogExpectedImprovement(model, best_f=train_y.max())
    bounds = torch.tensor([[-2.0], [2.0]])
    candidate, _ = optimize_acqf(
        EI,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    new_x = candidate
    new_y = f(new_x)

    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    plot_iteration(model, train_x, train_y, f, bounds, i)


print("Best value:", train_y.max())
