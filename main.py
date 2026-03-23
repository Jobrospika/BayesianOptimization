import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


#Config
BOUNDS = torch.tensor([[-2.0], [2.0]])
N_INIT = 3
N_ITER = 20


#Objective Function
def f(x):
    return torch.sin(3 * x) + x**2


#Normilization
def normalize(x, bounds):
    return (x - bounds[0]) / (bounds[1] - bounds[0])

def unnormalize(x, bounds):
    return x * (bounds[1] - bounds[0]) + bounds[0]


#Model
def train_model(train_x, train_y):
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

#Acquisition Function
def get_acquisition(model, train_y):
    return LogExpectedImprovement(model, best_f=train_y.max())


def optimize_acquisition(acq_func, bounds):
    candidate, _ = optimize_acqf(
        acq_func,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    candidate.clamp(0.0, 1.0)
    return candidate

#Plotting
def plot_iteration(model, train_x, train_y, iteration):
    plt.figure(1)
    plt.clf()

    x = torch.linspace(0, 1, 200).unsqueeze(-1)

    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

    plt.plot(x.numpy(), mean.numpy(), label="GP mean")
    plt.fill_between(
        x.squeeze().numpy(),
        (mean - 2 * std).squeeze().numpy(),
        (mean + 2 * std).squeeze().numpy(),
        alpha=0.3,
    )

    plt.scatter(train_x.numpy(), train_y.numpy(), color="red")

    plt.title(f"Iteration {iteration}")
    plt.legend()

    #Save Figures if necessary
    #plt.savefig(f"Plots/iteration_{iteration:02d}.png")

    plt.pause(2)


#Main Loop
def run_bayesian_optimization():
    train_x_orig = torch.linspace(BOUNDS[0].item(), BOUNDS[1].item(), N_INIT).unsqueeze(-1)
    train_y = f(train_x_orig)
    train_x = normalize(train_x_orig, BOUNDS)

    for i in range(N_ITER):
        model = train_model(train_x, train_y)

        acq = get_acquisition(model, train_y)
        candidate = optimize_acquisition(acq, torch.tensor([[0.0], [1.0]]))

        new_x = candidate
        new_y = f(unnormalize(new_x, BOUNDS))

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        plot_iteration(model, train_x, train_y, i)
        print(f"Iteration {i}: new point (normalized) = {new_x.squeeze().numpy()}")
        print(f"Iteration {i}: new point (original space) = {unnormalize(new_x, BOUNDS).squeeze().numpy()}")
        print(f"Iteration {i}: train_x.shape = {train_x.shape}, train_y.shape = {train_y.shape}")

    return train_x, train_y


#Run like a trackstar
def main():
    torch.set_default_dtype(torch.double)
    plt.ion()

    train_x, train_y = run_bayesian_optimization()

    print("Best value:", train_y.max())


if __name__ == "__main__":
    main()