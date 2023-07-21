def setup_ax(ax, title, xlim=(-1000, 1000), ylim=(-1000, 1000), aspect="equal", xlabel="x", ylabel="y"):
    ax.grid(which="both", linestyle="-", alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
