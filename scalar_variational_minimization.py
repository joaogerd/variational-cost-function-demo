#!/usr/bin/env python3
"""
Demonstração didática da minimização variacional em um problema escalar.

Este programa ilustra, da forma mais simples possível, um comportamento típico
observado em assimilação variacional de dados:

    J(x) = J_B(x) + J_O(x)

Quando a minimização parte do background x_b, é comum observar:

    J_O  diminui
    J_B  aumenta
    J    diminui

Isso não é uma falha do minimizador. O termo J_B começa no seu mínimo quando
x = x_b. Para aproximar o estado das observações, o minimizador precisa se
afastar do background; esse afastamento aumenta J_B, mas pode produzir uma
redução ainda maior em J_O e, portanto, reduzir a função custo total J.

Problema considerado
--------------------
O exemplo possui apenas uma variável de estado x e usa um operador de
observação identidade, H = 1. Assim,

    J_B(x) = 1/2 * (x - x_b)^2 / B
    J_O(x) = 1/2 * (y - x)^2 / R
    J(x)   = J_B(x) + J_O(x)

onde:

    x_b : background (primeira estimativa)
    y   : observação
    B   : variância do erro do background
    R   : variância do erro da observação

Neste script:

    x_b = 0
    y   = 10
    B   = 4
    R   = 1

A solução analítica é x_a = 8. Como R < B, a observação possui menor variância
que o background e, por isso, a análise fica mais próxima de y.

Minimizador
-----------
Para tornar o processo visível passo a passo, usamos gradiente descendente:

    x_{k+1} = x_k - alpha * grad J(x_k)

Esse algoritmo foi escolhido pela simplicidade didática. Sistemas reais de
assimilação usam algoritmos mais sofisticados, mas a interpretação da função
custo continua sendo a mesma.

Interpretação dos painéis
-------------------------
PAINEL ESQUERDO — espaço de estado

    As curvas J_B(x), J_O(x) e J(x) são funções FIXAS.

    O que se move é o estado corrente x_k. A linha vertical vermelha representa
    esse mesmo x_k em cada iteração. Os pontos sobre J_B, J_O e J mostram os
    valores das três funções avaliadas exatamente no mesmo estado.

    A bolinha vermelha sobre J(x) representa a posição atual do minimizador:

        (x_k, J(x_k))

    Além disso, um TRACEJADO VERMELHO liga as posições anteriores da bolinha
    vermelha. Esse tracejado é importante porque mostra que a trajetória do
    minimizador é DISCRETA, composta pelos pontos x_0, x_1, x_2, ...,
    e não um deslocamento contínuo ao longo da curva.

PAINEL DIREITO — história da minimização

    O eixo horizontal deixa de ser o estado x e passa a ser a iteração k.
    As curvas J_B(k), J_O(k) e J(k) são construídas progressivamente usando os
    valores calculados no painel esquerdo.

    Assim, os dois painéis mostram o MESMO processo sob duas perspectivas:

        esquerda -> onde está o minimizador no espaço de estado;
        direita  -> como os termos da função custo evoluem com as iterações.

Uso
---
Execução interativa:

    python scalar_variational_minimization.py

Execução mais lenta:

    python scalar_variational_minimization.py --pause 4

Salvar animação como GIF:

    python scalar_variational_minimization.py \
        --save-gif figures/scalar_variational_minimization.gif

Observação importante
---------------------
Este é propositalmente um exemplo escalar. Em problemas reais, x é um vetor de
alta dimensão, B e R são matrizes de covariância e H pode ser não linear. Ainda
assim, este exemplo preserva a interpretação fundamental do compromisso entre
background e observações.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================================
# 1. DEFINIÇÃO DO PROBLEMA DE ASSIMILAÇÃO
# ============================================================================
# O problema foi escolhido para ser o mais simples possível. Ainda assim, ele
# preserva todos os ingredientes conceituais importantes da minimização
# variacional: background, observação, função custo e balanço entre J_B e J_O.
# ============================================================================

xb = 0.0
"""Background (primeira estimativa do estado)."""

y = 10.0
"""Observação escalar assimilada."""

sigma_b = 2.0
"""Desvio-padrão do erro do background."""

sigma_o = 1.0
"""Desvio-padrão do erro da observação."""

B = sigma_b**2
"""Variância do erro do background."""

R = sigma_o**2
"""Variância do erro da observação."""


# ============================================================================
# 2. FUNÇÕES CUSTO
# ============================================================================
# Em um problema escalar com H = 1:
#
#   J_B(x) = 1/2 * (x - x_b)^2 / B
#   J_O(x) = 1/2 * (y - x)^2 / R
#   J(x)   = J_B(x) + J_O(x)
#
# J_B mede o custo associado a afastar a solução do background.
# J_O mede o custo associado a não reproduzir a observação.
# J    mede o compromisso total entre essas duas exigências.
# ============================================================================


def Jb(x):
    """Calcula o termo de background J_B(x).

    Parameters
    ----------
    x : float or numpy.ndarray
        Estado escalar no qual o custo será avaliado.

    Returns
    -------
    float or numpy.ndarray
        Valor de J_B no estado x.
    """
    return 0.5 * (x - xb) ** 2 / B


def Jo(x):
    """Calcula o termo observacional J_O(x).

    Como H = 1, o equivalente do modelo à observação é o próprio x.
    Assim, o desajuste observacional é simplesmente (y - x).
    """
    return 0.5 * (y - x) ** 2 / R


def J(x):
    """Calcula a função custo total J(x) = J_B(x) + J_O(x)."""
    return Jb(x) + Jo(x)


def grad_J(x):
    """Calcula o gradiente da função custo total.

    Para o problema escalar:

        dJ/dx = (x - x_b) / B + (x - y) / R

    O minimizador usa esse gradiente para atualizar x_k via gradiente
    descendente.
    """
    return (x - xb) / B + (x - y) / R


# ============================================================================
# 3. SOLUÇÃO ANALÍTICA
# ============================================================================
# A solução exata do problema escalar serve como referência para interpretar
# a convergência do minimizador numérico.
# ============================================================================

xa = (R * xb + B * y) / (B + R)
"""Análise exata do problema escalar."""


# ============================================================================
# 4. ROTINA DE MINIMIZAÇÃO
# ============================================================================
# Esta função gera toda a sequência x_0, x_1, ..., x_k.
# O histórico é armazenado porque será usado nos dois painéis:
#
#   - no painel esquerdo, para mostrar a trilha discreta da bolinha vermelha;
#   - no painel direito, para construir as curvas J_B(k), J_O(k) e J(k).
# ============================================================================


def run_minimization(alpha=0.5, n_iterations=10):
    """Executa gradiente descendente e armazena o histórico completo.

    Parameters
    ----------
    alpha : float, optional
        Tamanho do passo do gradiente descendente.
    n_iterations : int, optional
        Número total de iterações do minimizador.

    Returns
    -------
    list of dict
        Lista contendo, para cada iteração k:
        - k      : número da iteração
        - x      : estado x_k
        - Jb     : J_B(x_k)
        - Jo     : J_O(x_k)
        - J      : J(x_k)
        - grad   : grad J(x_k)
    """
    history = []
    x = xb

    for k in range(n_iterations + 1):
        history.append(
            {
                "k": k,
                "x": x,
                "Jb": Jb(x),
                "Jo": Jo(x),
                "J": J(x),
                "grad": grad_J(x),
            }
        )

        if k < n_iterations:
            x = x - alpha * grad_J(x)

    return history


# ============================================================================
# 5. CRIAÇÃO DA FIGURA
# ============================================================================
# O layout usa dois painéis lado a lado. Isso torna fácil comparar:
#
#   painel esquerdo -> geometria da função custo no espaço de estado;
#   painel direito  -> evolução das quantidades ao longo das iterações.
# ============================================================================


def create_figure():
    """Cria a figura e retorna os dois eixos principais."""
    fig, (ax_state, ax_iter) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        constrained_layout=True,
    )
    return fig, ax_state, ax_iter


# ============================================================================
# 6. DESENHO DE CADA QUADRO
# ============================================================================
# Esta é a função central do programa. Cada chamada desenha a figura
# correspondente a uma iteração específica do minimizador.
# ============================================================================


def draw_frame(fig, ax_state, ax_iter, history, frame):
    """Desenha o quadro correspondente a uma iteração.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura principal.
    ax_state : matplotlib.axes.Axes
        Eixo do painel esquerdo (espaço de estado).
    ax_iter : matplotlib.axes.Axes
        Eixo do painel direito (histórico por iteração).
    history : list of dict
        Histórico produzido por ``run_minimization``.
    frame : int
        Índice da iteração a ser desenhada.
    """
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

    # ========================================================================
    # PAINEL ESQUERDO — ESPAÇO DE ESTADO
    # ========================================================================
    # As curvas abaixo são FIXAS. Elas são funções do estado x.
    # O minimizador não as move. O que se move é o estado atual x_k.
    # ========================================================================

    line_jb, = ax_state.plot(xx, Jb(xx), linewidth=2.2, label=r"$J_B(x)$")
    line_jo, = ax_state.plot(xx, Jo(xx), linewidth=2.2, label=r"$J_O(x)$")
    ax_state.plot(xx, J(xx), linewidth=3.0, label=r"$J(x)=J_B(x)+J_O(x)$")

    # Referências fixas do problema:
    #   x_b -> posição do background
    #   y   -> posição da observação
    #   x_a -> posição da análise exata
    ax_state.axvline(xb, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(y, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(xa, linestyle=":", linewidth=2.0, alpha=0.85)

    # A linha vertical vermelha marca o estado atual x_k.
    ax_state.axvline(
        xk,
        color="red",
        linewidth=2.0,
        alpha=0.75,
        label=rf"estado atual $x_k={xk:.2f}$",
    )

    # ------------------------------------------------------------------------
    # TRAJETÓRIA DISCRETA DA BOLINHA VERMELHA
    # ------------------------------------------------------------------------
    # Este tracejado vermelho é importante didaticamente: ele mostra que a
    # minimização é composta por uma sequência discreta de iterações
    # (x_0, x_1, x_2, ...) e não por um deslocamento contínuo sobre a curva J.
    # ------------------------------------------------------------------------
    shown = history[: frame + 1]
    path_x = [d["x"] for d in shown]
    path_J = [d["J"] for d in shown]

    ax_state.plot(
        path_x,
        path_J,
        "o--",
        color="red",
        linewidth=1.4,
        markersize=4,
        alpha=0.65,
        zorder=8,
        label="trajetória discreta do minimizador",
    )

    # Ponto principal: bolinha vermelha indicando a posição atual do minimizador
    # sobre a função custo total J(x).
    ax_state.scatter(
        xk,
        jt,
        s=150,
        color="red",
        edgecolor="white",
        linewidth=1.4,
        zorder=10,
    )

    # Pontos auxiliares: valores de J_B(x_k) e J_O(x_k) avaliados no mesmo x_k.
    # Isso reforça a ideia de que os três termos são funções do MESMO estado.
    ax_state.scatter(
        xk,
        jb,
        s=95,
        color=line_jb.get_color(),
        edgecolor="white",
        linewidth=1.2,
        zorder=9,
    )
    ax_state.scatter(
        xk,
        jo,
        s=95,
        color=line_jo.get_color(),
        edgecolor="white",
        linewidth=1.2,
        zorder=9,
    )

    # Segmento vertical conectando visualmente J_B(x_k), J_O(x_k) e J(x_k).
    # Ele deixa claro que todos são avaliados na mesma abscissa x_k.
    ax_state.plot(
        [xk, xk],
        [min(jb, jo, jt), max(jb, jo, jt)],
        color="red",
        linewidth=1.0,
        alpha=0.35,
    )

    # Rótulos dos valores calculados no estado atual.
    ax_state.annotate(
        rf"$J_B(x_k)={jb:.2f}$",
        (xk, jb),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=line_jb.get_color(),
    )
    ax_state.annotate(
        rf"$J_O(x_k)={jo:.2f}$",
        (xk, jo),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=line_jo.get_color(),
    )
    ax_state.annotate(
        rf"$J(x_k)={jt:.2f}$",
        (xk, jt),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        color="red",
    )

    y_label = 53.0
    ax_state.text(xb, y_label, r"$x_b$", ha="center", va="center")
    ax_state.text(xa, y_label, r"$x_a$", ha="center", va="center")
    ax_state.text(y, y_label, r"$y$", ha="center", va="center")

    # Pequena caixa textual que muda com o andamento da minimização.
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
        0.025,
        0.57,
        explanation,
        transform=ax_state.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.90),
    )

    ax_state.set_xlim(-2, 12)
    ax_state.set_ylim(0, 56)
    ax_state.set_xlabel(r"Estado $x$")
    ax_state.set_ylabel("Valor da função custo")
    ax_state.set_title(
        "Painel esquerdo — espaço de estado\n"
        r"$x_k$ anda em passos discretos e é avaliado em $J_B$, $J_O$ e $J$"
    )
    ax_state.grid(alpha=0.25)
    ax_state.legend(loc="upper center", fontsize=8.7)

    # ========================================================================
    # PAINEL DIREITO — HISTÓRICO DA MINIMIZAÇÃO
    # ========================================================================
    # Aqui usamos exatamente o mesmo histórico, mas agora como função da
    # iteração k. As curvas são construídas progressivamente, ponto a ponto.
    # ========================================================================

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

    ax_iter.annotate(
        rf"$J_O={jo:.2f}$",
        (k, jo),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )
    ax_iter.annotate(
        rf"$J_B={jb:.2f}$",
        (k, jb),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )
    ax_iter.annotate(
        rf"$J={jt:.2f}$",
        (k, jt),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=10,
        color="red",
    )

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
        fontsize=15,
    )


# ============================================================================
# 7. IMPRESSÃO NO TERMINAL
# ============================================================================
# Além da visualização gráfica, o programa imprime cada passo do algoritmo no
# terminal. Isso ajuda a acompanhar explicitamente a natureza discreta do método.
# ============================================================================


def print_step(data, alpha):
    """Imprime no terminal os valores da iteração atual e o próximo passo."""
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


# ============================================================================
# 8. EXECUÇÃO INTERATIVA
# ============================================================================


def run_interactive(history, alpha, pause):
    """Exibe a animação em uma janela interativa."""
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


# ============================================================================
# 9. SALVAR GIF
# ============================================================================
# A mesma rotina draw_frame é usada tanto na visualização interativa quanto na
# criação do GIF. Portanto, qualquer melhoria visual introduzida na animação
# interativa aparece automaticamente no GIF exportado.
# ============================================================================


def save_gif(history, output, interval_ms):
    """Salva a animação como GIF.

    Parameters
    ----------
    history : list of dict
        Histórico gerado pelo minimizador.
    output : str or pathlib.Path
        Caminho do GIF a ser gravado.
    interval_ms : int
        Intervalo entre quadros em milissegundos.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_state, ax_iter = create_figure()

    def update(frame):
        draw_frame(fig, ax_state, ax_iter, history, frame)
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval_ms,
        repeat=True,
    )

    fps = max(1, round(1000 / interval_ms))
    animation.save(output, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"GIF salvo em: {output}")


# ============================================================================
# 10. INTERFACE DE LINHA DE COMANDO
# ============================================================================
# Permite ajustar velocidade, número de iterações e exportação para GIF.
# ============================================================================


def main():
    """Função principal da aplicação."""
    parser = argparse.ArgumentParser(
        description="Demonstração escalar da minimização variacional."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Tamanho do passo do gradiente descendente (default: 0.5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Número de iterações (default: 10).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Segundos entre quadros no modo interativo (default: 2).",
    )
    parser.add_argument(
        "--save-gif",
        metavar="ARQUIVO",
        default=None,
        help="Salva a animação como GIF em vez de abrir a janela.",
    )
    parser.add_argument(
        "--gif-interval",
        type=int,
        default=1200,
        help="Intervalo entre quadros do GIF em ms (default: 1200).",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations deve ser >= 1")
    if args.alpha <= 0:
        parser.error("--alpha deve ser > 0")
    if args.pause < 0:
        parser.error("--pause deve ser >= 0")
    if args.gif_interval <= 0:
        parser.error("--gif-interval deve ser > 0")

    history = run_minimization(alpha=args.alpha, n_iterations=args.iterations)

    if args.save_gif:
        save_gif(history, args.save_gif, args.gif_interval)
    else:
        run_interactive(history, args.alpha, args.pause)


if __name__ == "__main__":
    main()
