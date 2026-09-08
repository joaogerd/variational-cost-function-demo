#!/usr/bin/env python3
"""
Demonstração didática da minimização variacional em um problema escalar.

Este programa foi escrito para ilustrar, da forma mais simples possível, um
comportamento típico da assimilação variacional de dados:

    J(x) = J_B(x) + J_O(x)

Durante a minimização, quando o estado inicial é o background x_b, é comum
observar:

    J_O  diminui
    J_B  aumenta
    J    diminui

Isso não representa uma falha do minimizador. O termo J_B começa em seu mínimo
quando x = x_b. Para aproximar o estado das observações, o minimizador precisa
se afastar do background; esse afastamento aumenta J_B, mas pode produzir uma
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
de erro que o background e, por isso, a análise fica mais próxima de y.

Minimizador
-----------
Para tornar o processo visível passo a passo, usamos gradiente descendente:

    x_{k+1} = x_k - alpha * grad J(x_k)

Esse algoritmo foi escolhido pela simplicidade didática. Sistemas reais de
assimilação podem usar algoritmos muito mais sofisticados, mas a interpretação
da função custo continua sendo a mesma.

Interpretação dos painéis
-------------------------
PAINEL ESQUERDO — espaço de estado

    As curvas J_B(x), J_O(x) e J(x) são funções FIXAS.

    O que se move é o estado corrente x_k. A linha vertical vermelha representa
    esse mesmo x_k em cada iteração. Os pontos sobre J_B, J_O e J mostram os
    valores das três funções avaliadas exatamente no mesmo estado.

    A bolinha vermelha sobre J(x) representa a posição atual do minimizador:

        (x_k, J(x_k))

    Ela caminha do background em direção ao mínimo da função custo total.

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================================
# 1. DEFINIÇÃO DO PROBLEMA DE ASSIMILAÇÃO
# ============================================================================
#
# O objetivo é assimilar uma única observação y em uma única variável de
# estado x. O problema foi mantido escalar para que cada termo da função custo
# possa ser visto diretamente no gráfico.
#
# Em um sistema atmosférico real, x conteria muitas variáveis (temperatura,
# vento, umidade, pressão etc.) em milhões de pontos do domínio.
# ============================================================================

xb = 0.0       # x_b: background / primeira estimativa

y = 10.0       # y: observação

# Desvios-padrão dos erros associados às duas fontes de informação.
# Quanto menor o desvio-padrão, maior a confiança estatística naquela fonte.
sigma_b = 2.0  # erro-padrão do background
sigma_o = 1.0  # erro-padrão da observação

# No caso escalar, as matrizes de covariância B e R reduzem-se a variâncias.
B = sigma_b**2
R = sigma_o**2


# ============================================================================
# 2. TERMOS DA FUNÇÃO CUSTO
# ============================================================================


def Jb(x):
    """Calcula o termo de background J_B(x).

    Matemática
    ----------
        J_B(x) = 1/2 * (x - x_b)^2 / B

    Interpretação
    -------------
    J_B mede o custo estatístico de afastar o estado x do background x_b.

    Como a minimização começa em x = x_b, temos inicialmente:

        J_B(x_b) = 0

    Portanto, J_B já começa em seu menor valor possível. À medida que o
    minimizador move x para incorporar a observação, x se afasta de x_b e
    J_B tende a aumentar.

    Parameters
    ----------
    x : float ou numpy.ndarray
        Estado no qual J_B será avaliado.

    Returns
    -------
    float ou numpy.ndarray
        Valor do termo de background.
    """
    return 0.5 * (x - xb) ** 2 / B



def Jo(x):
    """Calcula o termo observacional J_O(x).

    Matemática
    ----------
    Para H = 1,

        J_O(x) = 1/2 * (y - x)^2 / R

    Interpretação
    -------------
    J_O mede o custo estatístico do desajuste entre a observação y e o estado
    x representado no espaço da observação.

    Neste exemplo H(x) = x, então o termo observacional é mínimo em x = y.
    À medida que o minimizador move x em direção à observação, J_O diminui.

    Parameters
    ----------
    x : float ou numpy.ndarray
        Estado no qual J_O será avaliado.

    Returns
    -------
    float ou numpy.ndarray
        Valor do termo observacional.
    """
    return 0.5 * (y - x) ** 2 / R



def J(x):
    """Calcula a função custo total J(x) = J_B(x) + J_O(x).

    O minimizador não procura minimizar J_B ou J_O isoladamente. Ele procura o
    estado que minimiza a soma dos dois termos, ou seja, o melhor compromisso
    estatístico entre background e observação.

    Parameters
    ----------
    x : float ou numpy.ndarray
        Estado no qual a função custo total será avaliada.

    Returns
    -------
    float ou numpy.ndarray
        Valor da função custo total.
    """
    return Jb(x) + Jo(x)



def grad_J(x):
    """Calcula o gradiente dJ/dx da função custo total.

    Para o problema escalar adotado aqui,

        dJ/dx = (x - x_b)/B + (x - y)/R

    O primeiro termo vem de J_B e tende a puxar a solução de volta para o
    background. O segundo vem de J_O e tende a puxar a solução em direção à
    observação.

    No mínimo da função custo,

        grad J(x_a) = 0,

    o que implica o equilíbrio entre as contribuições de background e
    observação.

    Parameters
    ----------
    x : float
        Estado corrente do minimizador.

    Returns
    -------
    float
        Gradiente da função custo no estado x.
    """
    return (x - xb) / B + (x - y) / R


# ============================================================================
# 3. SOLUÇÃO ANALÍTICA
# ============================================================================
#
# Em um problema linear e escalar com H = 1, podemos calcular diretamente a
# análise que minimiza J:
#
#              R x_b + B y
#       x_a = ----------------
#                 B + R
#
# Neste caso:
#
#       x_a = (1*0 + 4*10)/(4 + 1) = 8
#
# A solução analítica não é usada para conduzir a minimização. Ela serve como
# referência para verificar se o algoritmo iterativo converge para o ponto
# correto.
# ============================================================================

xa = (R * xb + B * y) / (B + R)


# ============================================================================
# 4. MINIMIZAÇÃO POR GRADIENTE DESCENDENTE
# ============================================================================


def run_minimization(alpha=0.5, n_iterations=10):
    """Executa a minimização e armazena o estado de cada iteração.

    O algoritmo utilizado é o gradiente descendente:

        x_{k+1} = x_k - alpha * grad J(x_k)

    A minimização começa exatamente no background:

        x_0 = x_b

    Em cada iteração armazenamos não apenas x_k, mas também J_B(x_k), J_O(x_k),
    J(x_k) e o gradiente. Esse histórico será usado pelo painel direito da
    animação.

    Parameters
    ----------
    alpha : float, optional
        Tamanho do passo do gradiente descendente. O valor padrão é 0.5.
        Neste exemplo quadrático ele produz uma convergência suave e fácil de
        acompanhar visualmente.

    n_iterations : int, optional
        Número de passos de minimização. O histórico contém também a condição
        inicial k=0, portanto possui n_iterations + 1 registros.

    Returns
    -------
    list of dict
        Histórico da minimização. Cada dicionário contém:

        ``k``
            Número da iteração.
        ``x``
            Estado corrente x_k.
        ``Jb``
            Valor J_B(x_k).
        ``Jo``
            Valor J_O(x_k).
        ``J``
            Valor total J(x_k).
        ``grad``
            Gradiente de J no estado x_k.
    """
    history = []

    # A condição inicial é o próprio background. Esse detalhe é essencial para
    # entender por que J_B começa em zero e cresce durante a minimização.
    x = xb

    for k in range(n_iterations + 1):
        # Avaliamos todos os termos da função custo no MESMO estado x_k.
        history.append({
            "k": k,
            "x": x,
            "Jb": Jb(x),
            "Jo": Jo(x),
            "J": J(x),
            "grad": grad_J(x),
        })

        if k < n_iterations:
            # Passo do gradiente descendente.
            #
            # Se grad J < 0, subtrair o gradiente move x para a direita.
            # Se grad J > 0, move x para a esquerda.
            # Em ambos os casos buscamos reduzir J.
            x = x - alpha * grad_J(x)

    return history


# ============================================================================
# 5. CRIAÇÃO DA FIGURA
# ============================================================================


def create_figure():
    """Cria os dois painéis usados na demonstração.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura principal.

    ax_state : matplotlib.axes.Axes
        Painel esquerdo. Mostra as funções no espaço de estado x.

    ax_iter : matplotlib.axes.Axes
        Painel direito. Mostra a evolução dos custos com a iteração k.
    """
    fig, (ax_state, ax_iter) = plt.subplots(
        1, 2, figsize=(15, 6.8), constrained_layout=True
    )
    return fig, ax_state, ax_iter


# ============================================================================
# 6. DESENHO DE UMA ITERAÇÃO
# ============================================================================


def draw_frame(fig, ax_state, ax_iter, history, frame):
    """Desenha uma etapa completa da minimização nos dois painéis.

    Esta função é o núcleo visual da demonstração. Um único índice ``frame``
    identifica o estado x_k atual. A partir dele desenhamos simultaneamente:

    1. no painel esquerdo, onde x_k está localizado nas funções J_B(x), J_O(x)
       e J(x);
    2. no painel direito, toda a história J_B(k), J_O(k) e J(k) acumulada até
       aquele mesmo instante.

    A ligação entre os painéis é, portanto:

        x_k  ->  J_B(x_k), J_O(x_k), J(x_k)  ->  ponto da iteração k

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura principal.

    ax_state, ax_iter : matplotlib.axes.Axes
        Eixos dos painéis esquerdo e direito.

    history : list of dict
        Histórico produzido por :func:`run_minimization`.

    frame : int
        Índice da iteração a ser mostrada.
    """
    data = history[frame]

    k = data["k"]
    xk = data["x"]
    jb = data["Jb"]
    jo = data["Jo"]
    jt = data["J"]
    grad = data["grad"]

    # Domínio usado apenas para desenhar as três funções suaves no painel
    # esquerdo. Ele não participa da minimização numérica.
    xx = np.linspace(-2.0, 12.0, 600)

    # Cada quadro é redesenhado do zero. Isso torna a implementação mais
    # simples e privilegia a clareza didática em vez de otimizações gráficas.
    ax_state.clear()
    ax_iter.clear()

    # ------------------------------------------------------------------------
    # PAINEL ESQUERDO — FUNÇÕES NO ESPAÇO DE ESTADO
    # ------------------------------------------------------------------------
    #
    # IMPORTANTE:
    # J_B(x), J_O(x) e J(x) são funções fixas. Elas NÃO se deslocam durante a
    # minimização. O elemento que se move é x_k.
    # ------------------------------------------------------------------------

    line_jb, = ax_state.plot(
        xx, Jb(xx), linewidth=2.2, label=r"$J_B(x)$"
    )
    line_jo, = ax_state.plot(
        xx, Jo(xx), linewidth=2.2, label=r"$J_O(x)$"
    )
    ax_state.plot(
        xx, J(xx), linewidth=3.0, label=r"$J(x)=J_B(x)+J_O(x)$"
    )

    # Três referências fixas no espaço de estado:
    #   x_b -> ponto de partida;
    #   y   -> observação;
    #   x_a -> mínimo analítico de J.
    ax_state.axvline(xb, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(y, linestyle="--", linewidth=1.4, alpha=0.55)
    ax_state.axvline(xa, linestyle=":", linewidth=2.0, alpha=0.85)

    # ------------------------------------------------------------------------
    # ESTADO CORRENTE x_k
    # ------------------------------------------------------------------------
    #
    # A linha vermelha é deliberadamente o principal elemento móvel do painel.
    # Ela representa UM ÚNICO estado x_k. Ao atravessar as três curvas, mostra
    # que J_B, J_O e J são todos avaliados no mesmo estado.
    # ------------------------------------------------------------------------

    ax_state.axvline(
        xk,
        color="red",
        linewidth=2.0,
        alpha=0.75,
        label=rf"estado atual $x_k={xk:.2f}$",
    )

    # Bolinha vermelha: posição atual do minimizador na função custo TOTAL.
    # Em outras palavras, representa o ponto (x_k, J(x_k)).
    ax_state.scatter(
        xk,
        jt,
        s=150,
        color="red",
        edgecolor="white",
        linewidth=1.4,
        zorder=10,
    )

    # Pontos auxiliares: componentes J_B e J_O avaliadas no mesmo x_k.
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

    # Segmento vertical entre os valores reforça visualmente a ideia:
    #
    #       um x_k -> três avaliações de custo.
    ax_state.plot(
        [xk, xk],
        [min(jb, jo, jt), max(jb, jo, jt)],
        color="red",
        linewidth=1.0,
        alpha=0.35,
    )

    # Valores numéricos da iteração atual junto aos três pontos.
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

    # Identificação das três posições conceitualmente importantes.
    y_label = 53.0
    ax_state.text(xb, y_label, r"$x_b$", ha="center", va="center")
    ax_state.text(xa, y_label, r"$x_a$", ha="center", va="center")
    ax_state.text(y, y_label, r"$y$", ha="center", va="center")

    # Mensagem dinâmica que descreve o significado físico/matemático da etapa.
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
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.90,
        ),
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

    # ------------------------------------------------------------------------
    # PAINEL DIREITO — HISTÓRIA DA MINIMIZAÇÃO
    # ------------------------------------------------------------------------
    #
    # Aqui o eixo horizontal NÃO representa mais o estado x. Ele representa o
    # número da iteração k. Os valores desenhados são exatamente aqueles
    # calculados no painel esquerdo para cada x_k visitado.
    #
    # Usamos somente history[:frame+1], portanto as curvas crescem passo a passo
    # e reproduzem visualmente a evolução de um gráfico real de convergência.
    # ------------------------------------------------------------------------

    shown = history[: frame + 1]

    iterations = [d["k"] for d in shown]
    values_jb = [d["Jb"] for d in shown]
    values_jo = [d["Jo"] for d in shown]
    values_j = [d["J"] for d in shown]

    ax_iter.plot(
        iterations,
        values_jo,
        marker="o",
        linewidth=2.4,
        label=r"$J_O(k)$",
    )
    ax_iter.plot(
        iterations,
        values_jb,
        marker="o",
        linewidth=2.4,
        label=r"$J_B(k)$",
    )
    ax_iter.plot(
        iterations,
        values_j,
        marker="o",
        linewidth=3.0,
        label=r"$J(k)$",
    )

    # Destaques da iteração atual. O ponto vermelho em J(k) corresponde à mesma
    # bolinha vermelha mostrada no painel esquerdo sobre J(x_k).
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

    # O título geral amarra os dois painéis ao mesmo estado x_k e mostra também
    # o gradiente. Quando grad J -> 0, estamos nos aproximando do mínimo.
    fig.suptitle(
        "Minimização variacional em um problema escalar\n"
        + rf"iteração {k}:  $x_k={xk:.3f}$,  $\nabla J={grad:.3f}$",
        fontsize=15,
    )


# ============================================================================
# 7. SAÍDA TEXTUAL DE CADA PASSO
# ============================================================================


def print_step(data, alpha):
    """Imprime no terminal os valores e a atualização da iteração atual.

    A saída textual complementa a animação. Ela permite acompanhar numericamente
    como um estado x_k produz os valores J_B, J_O e J e como o gradiente define
    o próximo estado x_{k+1}.

    Parameters
    ----------
    data : dict
        Registro de uma iteração do histórico.

    alpha : float
        Tamanho do passo usado pelo gradiente descendente.
    """
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
# 8. MODO INTERATIVO
# ============================================================================


def run_interactive(history, alpha, pause):
    """Reproduz a minimização lentamente em uma janela do Matplotlib.

    Cada quadro corresponde a uma iteração do histórico. O tempo de pausa
    controla quanto tempo o usuário tem para observar e explicar cada passo.

    Parameters
    ----------
    history : list of dict
        Histórico completo da minimização.

    alpha : float
        Tamanho do passo, usado apenas na impressão da atualização numérica.

    pause : float
        Tempo, em segundos, entre duas iterações consecutivas.
    """
    fig, ax_state, ax_iter = create_figure()
    plt.ion()

    for frame, data in enumerate(history):
        draw_frame(fig, ax_state, ax_iter, history, frame)
        print_step(data, alpha)
        fig.canvas.draw_idle()
        plt.pause(pause)

    plt.ioff()

    # Comparação final entre a solução analítica e o último estado obtido pelo
    # minimizador. Quanto mais iterações, mais próximo o valor numérico fica de
    # x_a.
    print("\n" + "=" * 68)
    print("RESULTADO")
    print("=" * 68)
    print(f"Background              x_b = {xb:.6f}")
    print(f"Observação                y = {y:.6f}")
    print(f"Análise analítica        x_a = {xa:.6f}")
    print(f"Último estado numérico       = {history[-1]['x']:.6f}")

    plt.show()


# ============================================================================
# 9. GERAÇÃO DO GIF
# ============================================================================


def save_gif(history, output, interval_ms):
    """Salva a mesma demonstração interativa em um arquivo GIF.

    O GIF é útil para documentação no GitHub, pois permite visualizar a
    minimização diretamente no README sem executar o script.

    Parameters
    ----------
    history : list of dict
        Histórico completo da minimização.

    output : str ou pathlib.Path
        Caminho do arquivo GIF de saída.

    interval_ms : int
        Intervalo entre dois quadros consecutivos, em milissegundos.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_state, ax_iter = create_figure()

    def update(frame):
        """Callback chamado pelo FuncAnimation para desenhar cada quadro."""
        draw_frame(fig, ax_state, ax_iter, history, frame)
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval_ms,
        repeat=True,
    )

    # PillowWriter trabalha em frames por segundo. O valor abaixo converte o
    # intervalo solicitado em uma taxa inteira mínima de 1 fps.
    fps = max(1, round(1000 / interval_ms))

    animation.save(
        output,
        writer=PillowWriter(fps=fps),
        dpi=105,
    )

    plt.close(fig)
    print(f"GIF salvo em: {output}")


# ============================================================================
# 10. INTERFACE DE LINHA DE COMANDO
# ============================================================================


def main():
    """Processa os argumentos e seleciona execução interativa ou geração GIF."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstração escalar da minimização variacional de "
            "J = J_B + J_O."
        )
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
        help="Número de iterações da minimização (default: 10).",
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

    # Validações simples evitam configurações sem sentido para a demonstração.
    if args.iterations < 1:
        parser.error("--iterations deve ser >= 1")
    if args.alpha <= 0:
        parser.error("--alpha deve ser > 0")
    if args.pause < 0:
        parser.error("--pause deve ser >= 0")
    if args.gif_interval <= 0:
        parser.error("--gif-interval deve ser > 0")

    # A minimização é executada uma única vez. Tanto o modo interativo quanto o
    # GIF usam exatamente o mesmo histórico e, portanto, mostram os mesmos
    # valores de x_k, J_B, J_O e J.
    history = run_minimization(args.alpha, args.iterations)

    if args.save_gif:
        save_gif(history, args.save_gif, args.gif_interval)
    else:
        run_interactive(history, args.alpha, args.pause)


if __name__ == "__main__":
    main()
