import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import io
import PIL


def visualize(input_tails_positions: torch.Tensor, exit_heads_positions: torch.Tensor):
    fig = plt.figure(facecolor="Black")
    ax = fig.add_subplot(111, projection="3d")  # 111 = 1 subplot in 1x1 grid

    # No background grid nor ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # style parameters
    fill_alpha = 0.06
    lines_alpha = 0.2
    dots_size = 15

    # Sphere template coordinates
    r = 1
    u = np.linspace(0, 2 * np.pi, 400)
    v = np.linspace(0, np.pi, 400)
    w = np.linspace(-r, r, 400)

    # Fill
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="lightgrey", edgecolor="none", alpha=fill_alpha)

    # Lines
    ## X line
    ax.plot(w, 0, 0, color="blue", linewidth=0.5, alpha=lines_alpha)
    ## Y line
    ax.plot(0, w, 0, color="green", linewidth=0.5, alpha=lines_alpha)
    ## Z line
    ax.plot(0, 0, w, color="red", linewidth=0.5, alpha=lines_alpha)

    # Circles
    ## YZ circle
    ax.plot(
        0, r * np.cos(u), r * np.sin(u), color="red", linewidth=0.5, alpha=lines_alpha
    )
    ## XY circle
    ax.plot(
        r * np.cos(u), r * np.sin(u), 0, color="green", linewidth=0.5, alpha=lines_alpha
    )
    ## XZ circle
    ax.plot(
        r * np.cos(u), 0, r * np.sin(u), color="blue", linewidth=0.5, alpha=lines_alpha
    )

    if exit_heads_positions.shape[-1] > 3:
        all_positions = torch.cat([exit_heads_positions, input_tails_positions], dim=0)
        U, S, _ = torch.pca_lowrank(all_positions, q=3)
        all_positions = U @ torch.diag(S)
        all_positions = F.normalize(all_positions, p=2, dim=-1).permute(1, 0)
        head_positions = all_positions[:, : exit_heads_positions.shape[0]]
        tail_positions = all_positions[:, exit_heads_positions.shape[0] :]
    else:
        head_positions = exit_heads_positions.T
        tail_positions = input_tails_positions.T

    # Heads
    ax.scatter(*head_positions.tolist(), color="purple", s=dots_size)
    # Tails
    ax.scatter(*tail_positions.tolist(), color="orange", s=dots_size)

    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    plt.close(fig)
    buf.seek(0)

    img = PIL.Image.open(buf)
    img = img.crop((155, 80, 510, 410))
    return np.asarray(img)
