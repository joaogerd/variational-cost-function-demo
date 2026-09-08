#!/usr/bin/env python3
"""Demonstração didática da minimização variacional em um problema escalar."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Problema escalar
xb = 0.0
y = 10.0
sigma_b = 2.0
sigma_o = 1.0
B = sigma_b**2
R = sigma_o**2


def Jb(x):
    return 0.5 * (x - xb) ** 2 / B


def Jo(x):
    return 0.5 * (y - x) ** 2 / R


def J(x):
    return Jb(x) + Jo(x)


def grad_J(x):
    return (x - xb) / B + (x - y) / R


xa = (R * xb + B * y) / (B + R)


def run_minimization(alpha=0.5, n_iterations=10):
    history = []
    x = xb
    for k in range(n_iterations + 1):
        history.append({
            "k": k,
            "x": x,
            "Jb": Jb(x),
            "Jo": Jo(x),
            "J": J(x),
            "grad": grad_J(x),
        })
        if k < n_iterations:
            x = x - alpha * grad_J(x)
    return history


def create_figure():
    fig, (ax_state, ax_iter) = plt.subplots(
        1, 2, figsize=(15, 6.8), constrained_layout=True
    )
    return fig, ax_state, ax_iter


def draw_frame(fig, ax_state, ax_iter, history, frame):
    data = history[frame]
    k = data["k"]
    xk = data["x"]
    jb = data["Jb"]
    jo = data["Jo"]
    jt = data["J"]
    grad = data["grad"]
    xx = np.linspace(-2.0, 12.0, 600)

    ax_state.clear()
    ax_iter.clear()

    # Painel esquerdo: funções fixas no espaço de estado
    line_jb, = ax_state.plot(xx, Jb(xx), linewidth=2.2, label=r"$J_B(x)$")
    line_jo, = ax_state.plot(xx, Jo(xx), linewidth=2.2, label=r"$J_O(x)$")
    ax_state.plot(xx, J(xx), linewidth=3.0, label=r"$J(x)=J_B(x)+J_O(x)$")

    # Referências fixas
    ax_state.axvline(xb, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(y, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(xa, linestyle=":", linewidth=2.0, alpha=0.85)

    # Estado atual do minimizador: linha vermelha e bolinha vermelha sobre J
    ax_state.axvline(
        xk, color="red", linewidth=2.0, alpha=0.75,
        label=rf"estado atual $x_k={xk:.2f}$"
    )
    ax_state.scatter(
        xk, jt, s=150, color="red", edgecolor="white",
        linewidth=1.4, zorder=10
    )

    # J_B e J_O avaliados no MESMO x_k
    ax_state.scatter(
        xk, jb, s=95, color=line_jb.get_color(),
        edgecolor="white", linewidth=1.2, zorder=9
    )
    ax_state.scatter(
        xk, jo, s=95, color=line_jo.get_color(),
        edgecolor="white", linewidth=1.2, zorder=9
    )

    # Segmento vertical reforça que os três valores usam o mesmo estado
    ax_state.plot(
        [xk, xk], [min(jb, jo, jt), max(jb, jo, jt)],
        color="red", linewidth=1.0, alpha=0.35
    )

    ax_state.annotate(
        rf"$J_B(x_k)={jb:.2f}$", (xk, jb),
        xytext=(10, 10), textcoords="offset points",
        fontsize=10, color=line_jb.get_color()
    )
    ax_state.annotate(
        rf"$J_O(x_k)={jo:.2f}$", (xk, jo),
        xytext=(10, 10), textcoords="offset points",
        fontsize=10, color=line_jo.get_color()
    )
    ax_state.annotate(
        rf"$J(x_k)={jt:.2f}$", (xk, jt),
        xytext=(10, -20), textcoords="offset points",
        fontsize=10, color="red"
    )

    y_label = 53.0
    ax_state.text(xb, y_label, r"$x_b$", ha="center", va="center")
    ax_state.text(xa, y_label, r"$x_a$", ha="center", va="center")
    ax_state.text(y, y_label, r"$y$", ha="center", va="center")

    if k == 0:
        explanation = (
            "INÍCIO\n\n"
            r"$x_0=x_b$" "\n"
            r"$J_B(x_b)=0$" "\n\n"
            r"$J_O$ ainda é grande."
        )
    elif abs(xk - xa) < 0.05:
        explanation = (
            "CONVERGÊNCIA\n\n"
            r"$\nabla J \approx 0$" "\n\n"
            "A análise atingiu o equilíbrio\n"
            "entre background e observação."
        )
    else:
        explanation = (
            rf"ITERAÇÃO {k}" "\n\n"
            r"$x_k$ afasta-se de $x_b$" "\n"
            r"$\Rightarrow J_B \uparrow$" "\n\n"
            r"$x_k$ aproxima-se de $y$" "\n"
            r"$\Rightarrow J_O \downarrow$"
        )

    ax_state.text(
        0.025, 0.57, explanation,
        transform=ax_state.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.90)
    )

    ax_state.set_xlim(-2, 12)
    ax_state.set_ylim(0, 56)
    ax_state.set_xlabel(r"Estado $x$")
    ax_state.set_ylabel("Valor da função custo")
    ax_state.set_title(
        "Painel esquerdo — espaço de estado\n"
        r"O mesmo $x_k$ é avaliado em $J_B$, $J_O$ e $J$"
    )
    ax_state.grid(alpha=0.25)
    ax_state.legend(loc="upper center", fontsize=9)

    # Painel direito: histórico até a iteração atual
    shown = history[: frame + 1]
    iterations = [d["k"] for d in shown]
    values_jb = [d["Jb"] for d in shown]
    values_jo = [d["Jo"] for d in shown]
    values_j = [d["J"] for d in shown]

    ax_iter.plot(iterations, values_jo, marker="o", linewidth=2.4, label=r"$J_O(k)$")
    ax_iter.plot(iterations, values_jb, marker="o", linewidth=2.4, label=r"$J_B(k)$")
    ax_iter.plot(iterations, values_j, marker="o", linewidth=3.0, label=r"$J(k)$")

    ax_iter.scatter(k, jo, s=110, zorder=10)
    ax_iter.scatter(k, jb, s=110, zorder=10)
    ax_iter.scatter(k, jt, s=130, color="red", zorder=11)

    ax_iter.annotate(rf"$J_O={jo:.2f}$", (k, jo), xytext=(8, 8),
                     textcoords="offset points", fontsize=10)
    ax_iter.annotate(rf"$J_B={jb:.2f}$", (k, jb), xytext=(8, 8),
                     textcoords="offset points", fontsize=10)
    ax_iter.annotate(rf"$J={jt:.2f}$", (k, jt), xytext=(8, -18),
                     textcoords="offset points", fontsize=10, color="red")

    ax_iter.set_xlim(0, history[-1]["k"])
    ax_iter.set_ylim(0, 55)
    ax_iter.set_xlabel("Iteração do minimizador")
    ax_iter.set_ylabel("Valor da função custo")
    ax_iter.set_title(
        "Painel direito — história da minimização\n"
        r"$J_O\downarrow,\quad J_B\uparrow,\quad J\downarrow$"
    )
    ax_iter.grid(alpha=0.25)
    ax_iter.legend(loc="upper right")

    fig.suptitle(
        "Minimização variacional em um problema escalar\n"
        + rf"iteração {k}:  $x_k={xk:.3f}$,  $\nabla J={grad:.3f}$",
        fontsize=15
    )


def print_step(data, alpha):
    k = data["k"]
    xk = data["x"]
    jb = data["Jb"]
    jo = data["Jo"]
    jt = data["J"]
    grad = data["grad"]

    print("=" * 68)
    print(f"ITERAÇÃO {k}")
    print("-" * 68)
    print(f"x_k        = {xk:10.5f}")
    print(f"J_B(x_k)   = {jb:10.5f}")
    print(f"J_O(x_k)   = {jo:10.5f}")
    print(f"J(x_k)     = {jt:10.5f}")
    print(f"grad J     = {grad:10.5f}")

    if abs(grad) > 1e-12:
        xnew = xk - alpha * grad
        print("\nPróximo passo:")
        print("x_{k+1} = x_k - alpha * grad J")
        print(f"x_{{k+1}} = {xk:.5f} - {alpha:.3f} * ({grad:.5f})")
        print(f"x_{{k+1}} = {xnew:.5f}")


def run_interactive(history, alpha, pause):
    fig, ax_state, ax_iter = create_figure()
    plt.ion()

    for frame, data in enumerate(history):
        draw_frame(fig, ax_state, ax_iter, history, frame)
        print_step(data, alpha)
        fig.canvas.draw_idle()
        plt.pause(pause)

    plt.ioff()
    print("\n" + "=" * 68)
    print("RESULTADO")
    print("=" * 68)
    print(f"Background              x_b = {xb:.6f}")
    print(f"Observação                y = {y:.6f}")
    print(f"Análise analítica        x_a = {xa:.6f}")
    print(f"Último estado numérico       = {history[-1]['x']:.6f}")
    plt.show()


def save_gif(history, output, interval_ms):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_state, ax_iter = create_figure()

    def update(frame):
        draw_frame(fig, ax_state, ax_iter, history, frame)
        return []

    animation = FuncAnimation(
        fig, update, frames=len(history), interval=interval_ms, repeat=True
    )
    fps = max(1, round(1000 / interval_ms))
    animation.save(output, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"GIF salvo em: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstração escalar da minimização variacional."
    )
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Tamanho do passo (default: 0.5).")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Número de iterações (default: 10).")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Segundos entre quadros no modo interativo (default: 2).")
    parser.add_argument("--save-gif", metavar="ARQUIVO", default=None,
                        help="Salva a animação como GIF em vez de abrir a janela.")
    parser.add_argument("--gif-interval", type=int, default=1200,
                        help="Intervalo entre quadros do GIF em ms (default: 1200).")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations deve ser >= 1")
    if args.alpha <= 0:
        parser.error("--alpha deve ser > 0")
    if args.pause < 0:
        parser.error("--pause deve ser >= 0")
    if args.gif_interval <= 0:
        parser.error("--gif-interval deve ser > 0")

    history = run_minimization(args.alpha, args.iterations)

    if args.save_gif:
        save_gif(history, args.save_gif, args.gif_interval)
    else:
        run_interactive(history, args.alpha, args.pause)


if __name__ == "__main__":
    main()
