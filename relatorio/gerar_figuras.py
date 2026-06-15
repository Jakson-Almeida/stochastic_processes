"""Gera todas as figuras e tabelas numéricas usadas no relatório.

Reproduz os cálculos do notebook notebooks/resolucao_trabalho.ipynb e salva:
  - figuras em relatorio/figuras/*.png
  - valores numéricos em relatorio/figuras/valores.tex (macros LaTeX)

Uso: python relatorio/gerar_figuras.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT if (ROOT / "dados_treino_branco.csv").exists() else ROOT / "data"
FIG = Path(__file__).resolve().parent / "figuras"
FIG.mkdir(exist_ok=True)

FS = 1.0
MAX_LAG = 50
CC_LAG = 60
NPERSEG = 512
M = 30
W = 50
ACF_LAG = 30
BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

valores = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("ok:", name)


def load(fname):
    df = pd.read_csv(DATA_DIR / fname)
    return df["n"].to_numpy(), df["u"].to_numpy(), df["y"].to_numpy()


def autocorr(x, max_lag):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    full = np.correlate(x, x, "full") / n
    mid = n - 1
    return full[mid : mid + max_lag + 1]


def crosscorr_yu(u, y, max_lag):
    u = np.asarray(u, float) - np.mean(u)
    y = np.asarray(y, float) - np.mean(y)
    n = len(u)
    full = np.correlate(y, u, "full") / n
    mid = n - 1
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, full[mid - max_lag : mid + max_lag + 1]


def fir_regressor(u, m):
    n = len(u)
    Phi = np.zeros((n, m))
    for k in range(m):
        Phi[k:, k] = u[: n - k]
    return Phi


def fit_fir(u, y, m):
    Phi = fir_regressor(u, m)
    h, *_ = np.linalg.lstsq(Phi[m - 1 :], y[m - 1 :], rcond=None)
    y_hat = Phi @ h
    e = y[m - 1 :] - y_hat[m - 1 :]
    r2 = 1 - np.sum(e ** 2) / np.sum((y[m - 1 :] - y[m - 1 :].mean()) ** 2)
    return h, y_hat, r2


# ---------------------------------------------------------------- dados
n_a, u_a, y_a = load("dados_treino_branco.csv")
n_b, u_b, y_b = load("dados_treino_colorido.csv")
n_v, u_v, y_v = load("dados_validacao_branco.csv")
n_c, u_c, y_c = load("dados_teste_anomalia.csv")

conf = 1.96 / np.sqrt(len(u_a))
lags_pos = np.arange(MAX_LAG + 1)

# ================================================================ TÓPICO A
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(n_a, u_a, lw=0.6)
ax[0].set_ylabel("u[n]")
ax[1].plot(n_a, y_a, lw=0.6, color="C1")
ax[1].set_ylabel("y[n]")
ax[1].set_xlabel("n")
save(fig, "A_sinais")

r_uu = autocorr(u_a, MAX_LAG)
r_yy = autocorr(y_a, MAX_LAG)
rho_uu, rho_yy = r_uu / r_uu[0], r_yy / r_yy[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, rho, t in zip(ax, [rho_uu, rho_yy], ["u[n]", "y[n]"]):
    a.stem(lags_pos, rho)
    a.axhline(conf, color="r", ls="--", lw=0.8)
    a.axhline(-conf, color="r", ls="--", lw=0.8)
    a.set_xlabel("lag k")
    a.set_ylabel(r"$\rho[k]$")
    a.set_title(t)
save(fig, "A_autocorr")
valores["AfracAutoU"] = f"{np.mean(np.abs(rho_uu[1:]) > conf):.3f}"
valores["AfracAutoY"] = f"{np.mean(np.abs(rho_yy[1:]) > conf):.3f}"

lags_cc, r_yu = crosscorr_yu(u_a, y_a, CC_LAG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.stem(lags_cc, r_yu)
ax.axhline(conf * u_a.std() * y_a.std(), color="r", ls="--", lw=0.8)
ax.axhline(-conf * u_a.std() * y_a.std(), color="r", ls="--", lw=0.8)
ax.set_xlabel("lag k")
ax.set_ylabel(r"$R_{yu}[k]$")
save(fig, "A_xcorr")
valores["ApicoXcorr"] = f"{lags_cc[np.argmax(np.abs(r_yu))]}"

f_u, Puu = signal.welch(u_a, fs=FS, nperseg=NPERSEG)
f_y, Pyy = signal.welch(y_a, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.semilogy(f_u, Puu, label=r"$S_{uu}$")
ax.semilogy(f_y, Pyy, label=r"$S_{yy}$", color="C1")
ax.set_xlabel("frequência (ciclos/amostra)")
ax.set_ylabel("PSD")
ax.legend()
save(fig, "A_psd")

f_c, Suy = signal.csd(u_a, y_a, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].semilogy(f_c, np.abs(Suy))
ax[0].set_ylabel(r"$|S_{yu}|$")
ax[1].plot(f_c, np.unwrap(np.angle(Suy)), color="C2")
ax[1].set_ylabel("fase (rad)")
ax[1].set_xlabel("frequência (ciclos/amostra)")
save(fig, "A_espectro_cruzado")

f_coh, coh = signal.coherence(u_a, y_a, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(f_coh, coh)
ax.axhline(0.9, color="r", ls="--", lw=0.8)
ax.set_ylim(0, 1.05)
ax.set_xlabel("frequência (ciclos/amostra)")
ax.set_ylabel(r"$\gamma^2_{uy}$")
save(fig, "A_coerencia")
valores["AcohMedia"] = f"{coh.mean():.3f}"
valores["AcohFrac"] = f"{np.mean(coh > 0.9):.3f}"

H1 = np.sqrt(Pyy / Puu)
H2 = Suy / Puu
fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
ax[0].plot(f_u, 20 * np.log10(H1), label=r"Método 1: $\sqrt{S_{yy}/S_{uu}}$")
ax[0].plot(f_c, 20 * np.log10(np.abs(H2)), "--", label=r"Método 2: $S_{yu}/S_{uu}$")
ax[0].set_ylabel("|H| (dB)")
ax[0].legend()
ax[1].plot(f_c, np.unwrap(np.angle(H2)), color="C2")
ax[1].set_ylabel("fase de H (rad)")
ax[1].set_xlabel("frequência (ciclos/amostra)")
save(fig, "A_resposta_frequencia")

h_fir, y_hat_a, r2_a = fit_fir(u_a, y_a, M)
valores["ArquadTreino"] = f"{r2_a:.4f}"
h_xcorr = r_yu[lags_cc >= 0][:M] / u_a.var()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].stem(np.arange(M), h_fir)
ax[0].set_xlabel("k")
ax[0].set_ylabel("h[k]")
ax[0].set_title("Resposta ao impulso (FIR)")
ax[1].stem(np.arange(M), h_fir, linefmt="C0-", markerfmt="C0o")
ax[1].plot(np.arange(M), h_xcorr, "C1x--", label=r"$R_{yu}/\sigma_u^2$")
ax[1].set_xlabel("k")
ax[1].set_title("FIR vs. correlação cruzada")
ax[1].legend()
save(fig, "A_fir_impulso")

# ordem vs R2
ordens = [5, 10, 15, 20, 30, 40, 60]
r2s = [fit_fir(u_a, y_a, m)[2] for m in ordens]
valores["ArquadMdez"] = f"{r2s[1]:.4f}"
valores["ArquadMcinco"] = f"{r2s[0]:.4f}"

seg = slice(0, 300)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(n_a[seg], y_a[seg], label="y[n]", lw=1.0)
ax.plot(n_a[seg], y_hat_a[seg], "--", label="ŷ[n] FIR", lw=1.0)
ax.set_xlabel("n")
ax.legend()
save(fig, "A_fir_ajuste")

w_fir, H_fir = signal.freqz(h_fir, worN=1024, fs=FS)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(f_u, 20 * np.log10(H1), alpha=0.6, label="Método 1")
ax.plot(f_c, 20 * np.log10(np.abs(H2)), alpha=0.6, label="Método 2")
ax.plot(w_fir, 20 * np.log10(np.abs(H_fir)), "k", label="FIR")
ax.set_xlabel("frequência (ciclos/amostra)")
ax.set_ylabel("|H| (dB)")
ax.legend()
save(fig, "A_fir_frequencia")

y_hat_v = np.convolve(u_v, h_fir)[: len(u_v)]
e_v = y_v[M:] - y_hat_v[M:]
r2_v = 1 - np.sum(e_v ** 2) / np.sum((y_v[M:] - y_v[M:].mean()) ** 2)
fit_v = 100 * (1 - np.linalg.norm(e_v) / np.linalg.norm(y_v[M:] - y_v[M:].mean()))
valores["ArquadVal"] = f"{r2_v:.4f}"
valores["Afitval"] = f"{fit_v:.1f}"
valores["ArmseVal"] = f"{np.sqrt(np.mean(e_v ** 2)):.4f}"
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(n_v[seg], y_v[seg], label="yval[n]", lw=1.0)
ax.plot(n_v[seg], y_hat_v[seg], "--", label="ŷval[n]", lw=1.0)
ax.set_xlabel("n")
ax.legend()
save(fig, "A_validacao")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(n_v[M:], e_v, lw=0.6)
ax[0].set_xlabel("n")
ax[0].set_ylabel("e[n]")
ax[0].set_title("Resíduos da validação")
r_ee = autocorr(e_v, 40)
ax[1].stem(np.arange(41), r_ee / r_ee[0])
ax[1].axhline(1.96 / np.sqrt(len(e_v)), color="r", ls="--", lw=0.8)
ax[1].axhline(-1.96 / np.sqrt(len(e_v)), color="r", ls="--", lw=0.8)
ax[1].set_xlabel("lag k")
ax[1].set_title("Autocorrelação dos resíduos")
save(fig, "A_residuos_validacao")

# estatísticas A
valores["AmediaU"] = f"{u_a.mean():.3f}"
valores["AvarU"] = f"{u_a.var():.3f}"
valores["AmediaY"] = f"{y_a.mean():.3f}"
valores["AvarY"] = f"{y_a.var():.3f}"
valores["AganhoDC"] = f"{h_fir.sum():.3f}"

# ================================================================ TÓPICO B
conf_b = 1.96 / np.sqrt(len(u_b))
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(n_b, u_b, lw=0.6)
ax[0].set_ylabel("u[n]")
ax[1].plot(n_b, y_b, lw=0.6, color="C1")
ax[1].set_ylabel("y[n]")
ax[1].set_xlabel("n")
save(fig, "B_sinais")

r_uu_b = autocorr(u_b, MAX_LAG)
r_yy_b = autocorr(y_b, MAX_LAG)
rho_uu_b, rho_yy_b = r_uu_b / r_uu_b[0], r_yy_b / r_yy_b[0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, rho, t in zip(ax, [rho_uu_b, rho_yy_b], ["u[n]", "y[n]"]):
    a.stem(lags_pos, rho)
    a.axhline(conf_b, color="r", ls="--", lw=0.8)
    a.axhline(-conf_b, color="r", ls="--", lw=0.8)
    a.set_xlabel("lag k")
    a.set_ylabel(r"$\rho[k]$")
    a.set_title(t)
save(fig, "B_autocorr")
valores["BfracAutoU"] = f"{np.mean(np.abs(rho_uu_b[1:]) > conf_b):.3f}"

lags_cc_b, r_yu_b = crosscorr_yu(u_b, y_b, CC_LAG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.stem(lags_cc_b, r_yu_b)
ax.axhline(conf_b * u_b.std() * y_b.std(), color="r", ls="--", lw=0.8)
ax.axhline(-conf_b * u_b.std() * y_b.std(), color="r", ls="--", lw=0.8)
ax.set_xlabel("lag k")
ax.set_ylabel(r"$R_{yu}[k]$")
save(fig, "B_xcorr")

f_u_b, Puu_b = signal.welch(u_b, fs=FS, nperseg=NPERSEG)
f_y_b, Pyy_b = signal.welch(y_b, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.semilogy(f_u_b, Puu_b, label=r"$S_{uu}$")
ax.semilogy(f_y_b, Pyy_b, label=r"$S_{yy}$", color="C1")
ax.set_xlabel("frequência (ciclos/amostra)")
ax.set_ylabel("PSD")
ax.legend()
save(fig, "B_psd")
valores["BrazaoPSD"] = f"{Puu_b.max() / Puu_b.min():.0f}"
valores["ArazaoPSD"] = f"{Puu.max() / Puu.min():.0f}"

f_c_b, Suy_b = signal.csd(u_b, y_b, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].semilogy(f_c_b, np.abs(Suy_b))
ax[0].set_ylabel(r"$|S_{yu}|$")
ax[1].plot(f_c_b, np.unwrap(np.angle(Suy_b)), color="C2")
ax[1].set_ylabel("fase (rad)")
ax[1].set_xlabel("frequência (ciclos/amostra)")
save(fig, "B_espectro_cruzado")

f_coh_b, coh_b = signal.coherence(u_b, y_b, fs=FS, nperseg=NPERSEG)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(f_coh_b, coh_b)
ax.axhline(0.9, color="r", ls="--", lw=0.8)
ax.set_ylim(0, 1.05)
ax.set_xlabel("frequência (ciclos/amostra)")
ax.set_ylabel(r"$\gamma^2_{uy}$")
save(fig, "B_coerencia")
valores["BcohMedia"] = f"{coh_b.mean():.3f}"
valores["BcohFrac"] = f"{np.mean(coh_b > 0.9):.3f}"

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].plot(lags_pos, rho_uu, "C0o-", label="branco", ms=3)
ax[0].plot(lags_pos, rho_uu_b, "C1s-", label="colorido", ms=3)
ax[0].axhline(conf, color="r", ls="--", lw=0.8)
ax[0].axhline(-conf, color="r", ls="--", lw=0.8)
ax[0].set_title(r"Autocorrelação de u")
ax[0].set_xlabel("lag k")
ax[0].legend()
ax[1].semilogy(f_u, Puu, label="branco")
ax[1].semilogy(f_u_b, Puu_b, label="colorido")
ax[1].set_title(r"PSD de u")
ax[1].set_xlabel("frequência (ciclos/amostra)")
ax[1].legend()
ax[2].plot(f_coh, coh, label="branco")
ax[2].plot(f_coh_b, coh_b, label="colorido")
ax[2].axhline(0.9, color="r", ls="--", lw=0.8)
ax[2].set_ylim(0, 1.05)
ax[2].set_title(r"Coerência")
ax[2].set_xlabel("frequência (ciclos/amostra)")
ax[2].legend()
save(fig, "B_comparacao")

# B.3 — FIR colorido vs referência
h_fir_b, _, _ = fit_fir(u_b, y_b, M)
w_ref, H_ref = signal.freqz(h_fir, worN=512, fs=FS)
w_col, H_col = signal.freqz(h_fir_b, worN=512, fs=FS)
err_db = 20 * np.log10(np.abs(H_col)) - 20 * np.log10(np.abs(H_ref))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(w_ref, 20 * np.log10(np.abs(H_ref)), label="FIR ref. (branco)", lw=1.5)
ax[0].plot(w_col, 20 * np.log10(np.abs(H_col)), "--", label="FIR (colorido)")
ax[0].set_xlabel("frequência (ciclos/amostra)")
ax[0].set_ylabel("|H| (dB)")
ax[0].legend()
ax2 = ax[1]
ax2.plot(w_col, np.abs(err_db), color="C3", label="|erro| (dB)")
ax2.set_xlabel("frequência (ciclos/amostra)")
ax2.set_ylabel("|erro| (dB)", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax2b = ax2.twinx()
ax2b.plot(f_coh_b, coh_b, color="C0", alpha=0.6)
ax2b.set_ylabel("coerência", color="C0")
ax2b.tick_params(axis="y", labelcolor="C0")
ax2b.set_ylim(0, 1.05)
save(fig, "B_erro_frequencia")

erro_faixa_b = []
for lo, hi in BANDS:
    m = (w_col >= lo) & (w_col < hi)
    mc = (f_coh_b >= lo) & (f_coh_b < hi)
    erro_faixa_b.append((f"[{lo:.1f}, {hi:.1f})", np.mean(np.abs(err_db[m])), coh_b[mc].mean()))
valores["BerroBaixa"] = f"{erro_faixa_b[0][1]:.2f}"
valores["BerroAlta"] = f"{erro_faixa_b[4][1]:.1f}"

# tabela comparação branco/colorido por faixa
band_rows = []
for lo, hi in BANDS:
    mw = (f_coh >= lo) & (f_coh < hi)
    mb = (f_coh_b >= lo) & (f_coh_b < hi)
    band_rows.append((f"[{lo:.1f}, {hi:.1f})", coh[mw].mean(), coh_b[mb].mean()))

valores["BvarY"] = f"{y_b.var():.3f}"
valores["BvarYbranco"] = f"{y_a.var():.3f}"

# ================================================================ TÓPICO C
y_hat_c = np.convolve(u_c, h_fir)[: len(u_c)]
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
ax[0].plot(n_c, u_c, lw=0.6)
ax[0].set_ylabel("u[n]")
ax[1].plot(n_c, y_c, lw=0.6, color="C1", label="y[n]")
ax[1].plot(n_c, y_hat_c, lw=0.6, color="C2", alpha=0.8, label="ŷ[n]")
ax[1].set_ylabel("y[n]")
ax[1].set_xlabel("n")
ax[1].legend(loc="upper right")
save(fig, "C_sinais")

e_c = y_c[M:] - y_hat_c[M:]
nn_c = n_c[M:]
baseline_std = e_v.std()
baseline_energy = e_v.var()
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(nn_c, e_c, lw=0.6)
ax.axhline(3 * baseline_std, color="r", ls="--", lw=0.8, label="±3·baseline")
ax.axhline(-3 * baseline_std, color="r", ls="--", lw=0.8)
ax.set_xlabel("n")
ax.set_ylabel("e[n]")
ax.legend(loc="upper left")
save(fig, "C_residuos")
valores["CbaselineStd"] = f"{baseline_std:.4f}"
valores["CvarRatio"] = f"{e_c.var() / baseline_energy:.1f}"

local_energy = np.convolve(e_c ** 2, np.ones(W) / W, mode="same")
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(nn_c, local_energy, label=f"energia local (W={W})")
ax.axhline(baseline_energy, color="g", ls="--", lw=1.0, label="baseline")
ax.set_xlabel("n")
ax.set_ylabel(r"$\langle e^2 \rangle$")
ax.legend()
save(fig, "C_energia_local")
valores["CenergiaRatio"] = f"{local_energy.max() / baseline_energy:.0f}"

thr_energy = 9 * baseline_energy
above = np.where(local_energy > thr_energy)[0]
n_change = int(nn_c[above[0]]) if len(above) else None
valores["CnChange"] = f"{n_change}"
mask_normal = nn_c < n_change
mask_anom = nn_c >= n_change

lags_e = np.arange(ACF_LAG + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for a, seg_e, t in zip(
    ax,
    [e_c, e_c[mask_normal], e_c[mask_anom]],
    ["Global", f"Normal (n<{n_change})", f"Anômalo (n>={n_change})"],
):
    r = autocorr(seg_e, ACF_LAG)
    cseg = 1.96 / np.sqrt(len(seg_e))
    a.stem(lags_e, r / r[0])
    a.axhline(cseg, color="r", ls="--", lw=0.8)
    a.axhline(-cseg, color="r", ls="--", lw=0.8)
    a.set_xlabel("lag k")
    a.set_title(t)
ax[0].set_ylabel(r"$\rho_{ee}[k]$")
save(fig, "C_autocorr_residuos")
r_norm = autocorr(e_c[mask_normal], ACF_LAG)
r_anom = autocorr(e_c[mask_anom], ACF_LAG)
valores["CautoNormal"] = f"{np.mean(np.abs((r_norm/r_norm[0])[1:]) > 1.96/np.sqrt(mask_normal.sum())):.3f}"
valores["CautoAnom"] = f"{np.mean(np.abs((r_anom/r_anom[0])[1:]) > 1.96/np.sqrt(mask_anom.sum())):.3f}"

# detecção: variância por blocos + impulsos
N_BLOCKS = 20
edges = np.linspace(0, len(e_c), N_BLOCKS + 1, dtype=int)
block_var, block_center = [], []
for i in range(N_BLOCKS):
    s = e_c[edges[i] : edges[i + 1]]
    block_var.append(s.var())
    block_center.append(nn_c[(edges[i] + edges[i + 1]) // 2])
imp_thr = 8 * baseline_std
imp_idx = np.where(np.abs(e_c) > imp_thr)[0]
valores["CnumImpulsos"] = f"{len(imp_idx)}"
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].bar(block_center, block_var, width=(len(e_c) / N_BLOCKS) * 0.9, alpha=0.7)
ax[0].axhline(baseline_energy, color="g", ls="--", label="baseline")
ax[0].axvline(n_change, color="k", ls=":", label=f"mudança ~n={n_change}")
ax[0].set_ylabel("variância do bloco")
ax[0].legend()
ax[1].plot(nn_c, e_c, lw=0.5, alpha=0.8)
ax[1].axhline(imp_thr, color="r", ls="--", lw=0.8)
ax[1].axhline(-imp_thr, color="r", ls="--", lw=0.8)
ax[1].plot(nn_c[imp_idx], e_c[imp_idx], "rx", ms=6, label=f"impulsos ({len(imp_idx)})")
ax[1].set_ylabel("e[n]")
ax[1].set_xlabel("n")
ax[1].legend(loc="upper left")
save(fig, "C_deteccao")

valores["CstdNormal"] = f"{e_c[mask_normal].std():.4f}"
valores["CstdAnom"] = f"{e_c[mask_anom].std():.4f}"
valores["CstdAnomRatio"] = f"{e_c[mask_anom].std() / baseline_std:.1f}"
valores["CmaxAnom"] = f"{np.max(np.abs(e_c[mask_anom])):.2f}"

# C.5 — y vs resíduos
y_seg = y_c[M:]
y_mean_normal = y_seg[mask_normal].mean()
baseline_y_var = y_seg[mask_normal].var()
local_energy_y = np.convolve((y_seg - y_mean_normal) ** 2, np.ones(W) / W, mode="same")
fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
ax[0].plot(nn_c, local_energy_y, label="energia local de y")
ax[0].axhline(baseline_y_var, color="g", ls="--", label="variância normal de y")
ax[0].axvline(n_change, color="k", ls=":", lw=0.8)
ax[0].set_ylabel(r"$\langle (y-\bar{y})^2 \rangle$")
ax[0].legend(fontsize=9)
ax[1].plot(nn_c, local_energy, label="energia local de e")
ax[1].axhline(baseline_energy, color="g", ls="--", label="baseline resíduos")
ax[1].axvline(n_change, color="k", ls=":", lw=0.8)
ax[1].set_ylabel(r"$\langle e^2 \rangle$")
ax[1].set_xlabel("n")
ax[1].legend(fontsize=9)
save(fig, "C_comparacao_y_residuo")

r_y_anom = autocorr(y_seg[mask_anom], ACF_LAG)
valores["CvarRatioY"] = f"{y_seg[mask_anom].var() / baseline_y_var:.1f}"
valores["CautoAnomY"] = f"{np.mean(np.abs((r_y_anom/r_y_anom[0])[1:]) > 1.96/np.sqrt(mask_anom.sum())):.2f}"
valores["CimpulsosY"] = f"{int(np.sum(np.abs(y_seg - y_mean_normal) > imp_thr))}"
valores["CpicoStdY"] = f"{np.max(np.abs(y_seg[mask_anom]-y_mean_normal))/y_seg[mask_normal].std():.1f}"
valores["CpicoStdE"] = f"{np.max(np.abs(e_c[mask_anom]))/baseline_std:.1f}"

# ---------------------------------------------------------------- macros + tabelas
lines = ["% Gerado por gerar_figuras.py - nao editar manualmente"]
for k, v in valores.items():
    lines.append(f"\\newcommand{{\\val{k}}}{{{v}}}")
(FIG.parent / "valores.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tabela_faixa(rows, fname, col2, col3):
    out = []
    for faixa, a, b in rows:
        out.append(f"{{{faixa}}} & {a:.3f} & {b:.3f}")
    (FIG / fname).write_text(" \\\\\n".join(out) + "\n", encoding="utf-8")


tabela_faixa(band_rows, "tab_coerencia_faixa.tex", "branco", "colorido")
# tabela erro por faixa (B.3)
out = []
for faixa, err, c in erro_faixa_b:
    out.append(f"{{{faixa}}} & {err:.2f} & {c:.3f}")
(FIG / "tab_erro_faixa.tex").write_text(" \\\\\n".join(out) + "\n", encoding="utf-8")

print("\nFiguras e valores gerados em", FIG.parent)
