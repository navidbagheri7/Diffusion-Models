"""Small empirical OT problems. Costs use c(x,y)=||x-y||^2/2."""

import torch
from scipy.optimize import linear_sum_assignment


def exact_pairing(source, target):
    """Uniform equal-size empirical OT via a Hungarian assignment (CPU O(n^3))."""
    if len(source) != len(target):
        raise ValueError("Assignment requires equal empirical sample counts")
    _, columns = linear_sum_assignment(torch.cdist(source, target).square().detach().cpu().numpy())
    return target[torch.as_tensor(columns, device=target.device)]


def conditional_pairing(source, target, labels):
    """Assign endpoints within classes, preserving each class's source marginal.

    Labels refer to the source rows. Here training uses data as ``source`` and
    independent Gaussian noise as ``target``. A global assignment would make
    class-conditional noise depend on the geometry of the data classes.
    """
    if len(source) != len(target) or len(labels) != len(source):
        raise ValueError("Conditional assignment requires equal sample/label counts")
    paired = torch.empty_like(target)
    for label in labels.unique():
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        paired[indices] = exact_pairing(source[indices], target[indices])
    return paired


def sinkhorn_plan(source, target, epsilon=0.5, iterations=1000, tolerance=1e-7):
    """Log-domain Sinkhorn; reports a failure rather than silently bad marginals.

    min_pi <pi,C> + epsilon KL(pi || a tensor b), fixed uniform marginals.
    Equivalently the reference kernel is exp(-||x-y||^2/(2 epsilon)).
    """
    if epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn requires positive regularization and iteration count")
    log_kernel = -torch.cdist(source.double(), target.double()).square() / (2 * epsilon)
    log_a = torch.full(
        (len(source),),
        -torch.log(torch.tensor(float(len(source)))).item(),
        device=source.device,
        dtype=torch.float64,
    )
    log_b = torch.full(
        (len(target),),
        -torch.log(torch.tensor(float(len(target)))).item(),
        device=source.device,
        dtype=torch.float64,
    )
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for index in range(iterations):
        u = log_a - torch.logsumexp(log_kernel + v[None], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
        if index % 20 == 0 or index == iterations - 1:
            plan = (log_kernel + u[:, None] + v[None]).exp()
            error = max(
                (plan.sum(1) - log_a.exp()).abs().max().item(),
                (plan.sum(0) - log_b.exp()).abs().max().item(),
            )
            if error < tolerance:
                return plan.to(source.dtype)
    raise RuntimeError(f"Sinkhorn did not converge: marginal residual {error:.3g}")


def sample_coupling(source, target, plan, n, generator=None):
    index = torch.multinomial(plan.flatten(), n, replacement=True, generator=generator)
    return source[index // len(target)], target[index % len(target)]
