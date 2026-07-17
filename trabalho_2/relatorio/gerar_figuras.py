"""Gera figuras, tabelas e valores numéricos do relatório do Trabalho 2.

Reproduz os cálculos do notebook notebooks/resolucao_trabalho.ipynb.

Uso: python trabalho_2/relatorio/gerar_figuras.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz, inv

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 11})

ROOT = Path(__file__).resolve().parent.parent  # trabalho_2/
DATA_DIR = ROOT / "dados"
FIG = Path(__file__).resolve().parent / "figuras"
FIG.mkdir(exist_ok=True)

FS = 1000.0
N_WIN = 256
N_TMPL = 64
MAX_LAG = 40
NFFT = 256


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_template():
    df = pd.read_csv(DATA_DIR / "template_pulso.csv")
    s = df["s"].to_numpy(dtype=float)
    assert len(s) == N_TMPL
    return s


def load_windows(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    ids = np.sort(df["record_id"].unique())
    Y = np.stack(
        [df.loc[df["record_id"] == i, "y"].to_numpy(dtype=float) for i in ids]
    )
    assert Y.shape[1] == N_WIN
    return Y


def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("record_id").reset_index(drop=True)


def biased_autocorr(x, max_lag):
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    full = np.correlate(x, x, mode="full") / n
    mid = n - 1
    return full[mid : mid + max_lag + 1]


# ---------------------------------------------------------------------------
# Signal ops
# ---------------------------------------------------------------------------

def true_pulse(s, n0, A, n_win=N_WIN):
    y = np.zeros(n_win)
    if A == 0 or n0 < 0:
        return y
    end = min(n_win, n0 + len(s))
    y[n0:end] = A * s[: end - n0]
    return y


def place_template(s, n0, n_win=N_WIN):
    v = np.zeros(n_win)
    end = min(n_win, n0 + len(s))
    if n0 < 0 or n0 >= n_win:
        return v
    v[n0:end] = s[: end - n0]
    return v


def matched_filter(y, s):
    """Correlação cruzada; retorna saída e n0 = argmax (início do template)."""
    # correlate(y, s)[k] ~ sum y[n] s[n-k]; pico em n0
    out = signal.correlate(y, s, mode="valid")  # length N_WIN - N_TMPL + 1
    k = int(np.argmax(out))
    return out, k, float(out[k])


def apply_wiener_freq(y, H):
    Y = np.fft.rfft(y, n=NFFT)
    return np.fft.irfft(H * Y, n=NFFT)[: len(y)]


def blue_amplitude(y, s_aligned, Cinv=None):
    if np.allclose(s_aligned, 0):
        return 0.0
    if Cinv is None:
        num = float(np.dot(s_aligned, y))
        den = float(np.dot(s_aligned, s_aligned))
    else:
        Cs = Cinv @ s_aligned
        num = float(np.dot(Cs, y))
        den = float(np.dot(Cs, s_aligned))
    return num / den if den > 0 else 0.0


def roc_curve_scores(y_true, scores):
    order = np.argsort(-scores)
    y_true = np.asarray(y_true)[order]
    scores = np.asarray(scores)[order]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    prev = None
    for yt, sc in zip(y_true, scores):
        if prev is None or sc != prev:
            tpr.append(tp / P if P else 0.0)
            fpr.append(fp / N if N else 0.0)
            prev = sc
        if yt == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / P if P else 0.0)
    fpr.append(fp / N if N else 0.0)
    tpr.append(1.0)
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def auc_trapz(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def detection_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def savefig(name):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path.name)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

s = load_template()
Y_noise = load_windows(DATA_DIR / "ruido_treino.csv")
Y_train = load_windows(DATA_DIR / "sinais_treino.csv")
Y_test = load_windows(DATA_DIR / "sinais_teste.csv")
lab_train = load_labels(DATA_DIR / "rotulos_treino.csv")
lab_test = load_labels(DATA_DIR / "rotulos_teste.csv")

assert Y_noise.shape == (300, N_WIN)
assert Y_train.shape == (500, N_WIN)
assert Y_test.shape == (250, N_WIN)
assert len(lab_train) == 500 and len(lab_test) == 250

# ---------------------------------------------------------------------------
# 1. Noise analysis
# ---------------------------------------------------------------------------

noise_flat = Y_noise.ravel()
mu_r = float(noise_flat.mean())
var_r = float(noise_flat.var(ddof=0))
std_r = float(np.sqrt(var_r))

# mean autocorr across windows
lags = np.arange(MAX_LAG + 1)
R_list = [biased_autocorr(Y_noise[i], MAX_LAG) for i in range(len(Y_noise))]
R_mean = np.mean(R_list, axis=0)
R_norm = R_mean / R_mean[0]
# fraction of lags 1..MAX_LAG outside white-noise band (approx 2/sqrt(N))
band = 2 / np.sqrt(N_WIN)
outside = float(np.mean(np.abs(R_norm[1:]) > band))

f_psd, psd_r = signal.welch(noise_flat, fs=FS, nperseg=128, noverlap=64)
psd_ratio = float(psd_r.max() / psd_r.min())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].stem(lags, R_norm, basefmt=" ")
axes[0].axhline(band, color="r", ls="--", lw=0.8, label=rf"$\pm 2/\sqrt{{N}}$")
axes[0].axhline(-band, color="r", ls="--", lw=0.8)
axes[0].set_xlabel("lag k")
axes[0].set_ylabel(r"$R_{rr}[k]/R_{rr}[0]$")
axes[0].set_title("Autocorrelação do ruído (média)")
axes[0].legend(fontsize=9)
axes[1].semilogy(f_psd, psd_r)
axes[1].set_xlabel("Frequência (Hz)")
axes[1].set_ylabel("PSD")
axes[1].set_title("PSD do ruído (Welch)")
savefig("01_ruido_autocorr_psd.png")

# Toeplitz covariance from mean autocorr (for colored BLUE)
# Use more lags for Cov: up to N_WIN-1 truncated via biased estimate on pooled
R_full = biased_autocorr(noise_flat - mu_r, N_WIN - 1)
C = toeplitz(R_full)
# regularize
eps = 1e-6 * np.trace(C) / N_WIN
C = C + eps * np.eye(N_WIN)
Cinv = inv(C)

noise_is_colored = outside > 0.05 or psd_ratio > 3.0

# ---------------------------------------------------------------------------
# 2. Template
# ---------------------------------------------------------------------------

t_s = np.arange(N_TMPL) / FS
S_f = np.fft.rfft(s, n=NFFT)
f_s = np.fft.rfftfreq(NFFT, d=1 / FS)
S_pow = np.abs(S_f) ** 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(t_s * 1000, s)
axes[0].set_xlabel("Tempo (ms)")
axes[0].set_ylabel("s[n]")
axes[0].set_title("Template do pulso")
axes[1].semilogy(f_s, S_pow / S_pow.max(), label="|S(f)|² (norm.)")
# resample noise PSD onto f_s for overlay
psd_interp = np.interp(f_s, f_psd, psd_r)
axes[1].semilogy(f_s, psd_interp / psd_interp.max(), label="PSD ruído (norm.)", alpha=0.8)
axes[1].set_xlabel("Frequência (Hz)")
axes[1].set_ylabel("Potência normalizada")
axes[1].set_title("Espectro do pulso vs ruído")
axes[1].legend(fontsize=9)
savefig("02_template_espectro.png")

f_peak = float(f_s[np.argmax(S_pow)])
# energy in low band 0-100 Hz
band_mask = f_s <= 100
energy_low = float(S_pow[band_mask].sum() / S_pow.sum())

# ---------------------------------------------------------------------------
# 3. Wiener filter (frequency domain)
# ---------------------------------------------------------------------------

# Signal spectrum model: template spectrum scaled by E[A^2] from train events
A_typ = float((lab_train.loc[lab_train.event_present == 1, "amplitude_A"] ** 2).mean())
# Place template at mid-window for spectral shape of signal component
s_mid = place_template(s, (N_WIN - N_TMPL) // 2)
Sxx = np.abs(np.fft.rfft(s_mid, n=NFFT)) ** 2 * A_typ
# Noise spectrum in the SAME convention (mean periodogram of the noise windows),
# so that Sxx and Nxx are directly comparable and H is properly scaled.
Nxx = np.mean(np.abs(np.fft.rfft(Y_noise, n=NFFT, axis=1)) ** 2, axis=0)
# Wiener: H = Sxx / (Sxx + Nxx)
H_w = Sxx / (Sxx + Nxx + 1e-18)

Y_train_w = np.stack([apply_wiener_freq(y, H_w) for y in Y_train])
Y_noise_w = np.stack([apply_wiener_freq(y, H_w) for y in Y_noise])
Y_test_w = np.stack([apply_wiener_freq(y, H_w) for y in Y_test])

# example plots
idx_pos = int(lab_train.index[lab_train.event_present == 1][0])
idx_neg = int(lab_train.index[lab_train.event_present == 0][0])
n_axis = np.arange(N_WIN)
t0p = int(lab_train.loc[idx_pos, "t0_sample"])
Ap = float(lab_train.loc[idx_pos, "amplitude_A"])

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
axes[0, 0].plot(n_axis, Y_train[idx_pos], lw=0.8)
axes[0, 0].axvline(t0p, color="r", ls="--", label=f"n0={t0p}")
axes[0, 0].set_title("Com pulso — bruto")
axes[0, 0].legend(fontsize=8)
axes[0, 1].plot(n_axis, Y_train_w[idx_pos], lw=0.8, color="C2")
axes[0, 1].axvline(t0p, color="r", ls="--")
axes[0, 1].set_title("Com pulso — Wiener")
axes[1, 0].plot(n_axis, Y_train[idx_neg], lw=0.8)
axes[1, 0].set_title("Só ruído — bruto")
axes[1, 0].set_xlabel("n")
axes[1, 1].plot(n_axis, Y_train_w[idx_neg], lw=0.8, color="C2")
axes[1, 1].set_title("Só ruído — Wiener")
axes[1, 1].set_xlabel("n")
savefig("03_wiener_exemplos.png")

# ---------------------------------------------------------------------------
# Pipelines: scores, n0, amplitude, reconstruction
# ---------------------------------------------------------------------------

def run_pipeline(Y, labels, use_wiener=False, use_matched=True, blue_colored=False):
    """Retorna dicts com scores, n0_hat, A_hat, y_hat, metrics pieces."""
    Y_in = Y
    if use_wiener:
        # precomputed maps by identity of array
        if Y is Y_train or np.shares_memory(Y, Y_train) or Y.shape == Y_train.shape and np.allclose(Y[0, :5], Y_train[0, :5]):
            Y_in = Y_train_w if Y.shape[0] == Y_train.shape[0] else Y_test_w
        # safer: check length
        if Y.shape[0] == Y_train.shape[0]:
            Y_in = Y_train_w
        elif Y.shape[0] == Y_test.shape[0]:
            Y_in = Y_test_w
        elif Y.shape[0] == Y_noise.shape[0]:
            Y_in = Y_noise_w
        else:
            Y_in = np.stack([apply_wiener_freq(y, H_w) for y in Y])

    n = len(Y_in)
    scores = np.zeros(n)
    n0_hat = np.zeros(n, dtype=int)
    A_hat = np.zeros(n)
    y_hat = np.zeros_like(Y_in)
    Cinv_use = Cinv if blue_colored else None

    for i in range(n):
        y = Y_in[i]
        if use_matched:
            out, n0, sc = matched_filter(y, s)
            scores[i] = sc
            n0_hat[i] = n0
        else:
            # raw: score = max |y| / or energy; n0 from max correlation with template anyway for fair amp
            out, n0, sc = matched_filter(y, s)
            scores[i] = float(np.max(np.abs(y)))  # detection on raw amplitude
            n0_hat[i] = n0
        sal = place_template(s, int(n0_hat[i]))
        # BLUE on original y (not Wiener) for amplitude when comparing BLUE; for reconstruction use estimated pulse
        y_for_blue = Y[i]  # always estimate A on original observation
        A_hat[i] = blue_amplitude(y_for_blue, sal, Cinv_use)
        y_hat[i] = A_hat[i] * sal

    return dict(scores=scores, n0_hat=n0_hat, A_hat=A_hat, y_hat=y_hat, Y_proc=Y_in)


def rmse_waveform(Y_hat, labels, s_tmpl):
    errs = []
    for i, row in labels.iterrows():
        if int(row.event_present) != 1:
            continue
        true = true_pulse(s_tmpl, int(row.t0_sample), float(row.amplitude_A))
        errs.append(np.sqrt(np.mean((Y_hat[i] - true) ** 2)))
    return float(np.mean(errs)) if errs else np.nan


def rmse_amp(A_hat, labels, mask=None):
    m = labels.event_present.to_numpy() == 1
    if mask is not None:
        m = m & mask
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((A_hat[m] - labels.amplitude_A.to_numpy()[m]) ** 2)))


def n0_error(n0_hat, labels, mask=None):
    m = labels.event_present.to_numpy() == 1
    if mask is not None:
        m = m & mask
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(n0_hat[m] - labels.t0_sample.to_numpy()[m])))


# Pipelines on train
pipes = {
    "bruto": dict(use_wiener=False, use_matched=False, blue_colored=False),
    "wiener": dict(use_wiener=True, use_matched=False, blue_colored=False),
    "casado": dict(use_wiener=False, use_matched=True, blue_colored=False),
    "wiener_casado": dict(use_wiener=True, use_matched=True, blue_colored=False),
    "blue_branco": dict(use_wiener=False, use_matched=True, blue_colored=False),
    "blue_colorido": dict(use_wiener=False, use_matched=True, blue_colored=True),
}

# Note: bruto/wiener use max|y| as detection score; casado variants use matched max
# blue_* same detection as casado but amplitude via BLUE (already in run_pipeline)

train_res = {}
for name, kw in pipes.items():
    train_res[name] = run_pipeline(Y_train, lab_train, **kw)

# Waveform reconstruction RMSE: for bruto use zero? Better: estimated pulse A*s(n0)
# For wiener reconstruction of waveform: use Wiener output truncated? Enunciado wants estimated impulsive signal.
# We use A_hat * s[n-n0] for all methods that estimate A; for "wiener" as waveform estimator use Wiener output as y_hat alternative

# Additional: wiener as waveform estimate directly
rmse_w_direct = []
for i, row in lab_train.iterrows():
    if int(row.event_present) != 1:
        continue
    true = true_pulse(s, int(row.t0_sample), float(row.amplitude_A))
    rmse_w_direct.append(np.sqrt(np.mean((Y_train_w[i] - true) ** 2)))
rmse_wiener_wave = float(np.mean(rmse_w_direct))

rmse_bruto_wave = []
for i, row in lab_train.iterrows():
    if int(row.event_present) != 1:
        continue
    true = true_pulse(s, int(row.t0_sample), float(row.amplitude_A))
    rmse_bruto_wave.append(np.sqrt(np.mean((Y_train[i] - true) ** 2)))
rmse_bruto_wave = float(np.mean(rmse_bruto_wave))

# ---------------------------------------------------------------------------
# 4. Matched filter diagnostics
# ---------------------------------------------------------------------------

res_c = train_res["casado"]
mask_pos = lab_train.event_present.to_numpy() == 1
mask_neg = ~mask_pos
n0_err_med = float(np.median(np.abs(res_c["n0_hat"][mask_pos] - lab_train.t0_sample.to_numpy()[mask_pos])))
n0_err_mean = float(np.mean(np.abs(res_c["n0_hat"][mask_pos] - lab_train.t0_sample.to_numpy()[mask_pos])))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(res_c["scores"][mask_neg], bins=30, alpha=0.7, label="sem pulso", density=True)
axes[0].hist(res_c["scores"][mask_pos], bins=30, alpha=0.7, label="com pulso", density=True)
axes[0].set_xlabel("máx. filtro casado")
axes[0].set_title("Separação das estatísticas (treino)")
axes[0].legend()
err_n0 = np.abs(res_c["n0_hat"][mask_pos] - lab_train.t0_sample.to_numpy()[mask_pos])
axes[1].hist(err_n0, bins=30, color="C1")
axes[1].set_xlabel(r"$|\hat{n}_0 - n_0|$")
axes[1].set_title("Erro de posição (casado, treino)")
savefig("04_casado_separacao.png")

# example matched output
out_ex, n0_ex, sc_ex = matched_filter(Y_train[idx_pos], s)
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
axes[0].plot(Y_train[idx_pos], lw=0.8)
axes[0].axvline(t0p, color="r", ls="--", label="n0 verdadeiro")
axes[0].axvline(n0_ex, color="g", ls=":", label="n0 estimado")
axes[0].legend(fontsize=8)
axes[0].set_title("Janela com pulso")
axes[1].plot(out_ex, lw=0.8)
axes[1].axvline(n0_ex, color="g", ls=":")
axes[1].set_title("Saída do filtro casado")
axes[1].set_xlabel("lag / n0 candidato")
savefig("05_casado_exemplo.png")

# ---------------------------------------------------------------------------
# 5. BLUE amplitude
# ---------------------------------------------------------------------------

A_true = lab_train.amplitude_A.to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name, title in zip(
    axes,
    ["blue_branco", "blue_colorido"],
    ["BLUE ruído branco", "BLUE ruído colorido"],
):
    Ah = train_res[name]["A_hat"]
    ax.scatter(A_true[mask_pos], Ah[mask_pos], s=12, alpha=0.6)
    lim = [0, max(A_true[mask_pos].max(), Ah[mask_pos].max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("A verdadeira")
    ax.set_ylabel(r"$\hat{A}$")
    ax.set_title(title)
savefig("06_blue_scatter.png")

# naive baseline: max(y)
A_naive = np.array([float(np.max(y)) for y in Y_train])
rmse_naive = float(np.sqrt(np.mean((A_naive[mask_pos] - A_true[mask_pos]) ** 2)))
rmse_blue_w = rmse_amp(train_res["blue_branco"]["A_hat"], lab_train)
rmse_blue_c = rmse_amp(train_res["blue_colorido"]["A_hat"], lab_train)

# ---------------------------------------------------------------------------
# 6. Threshold ~1% FA on noise-only (matched filter on raw noise)
# ---------------------------------------------------------------------------

scores_noise = np.array([matched_filter(y, s)[2] for y in Y_noise])
scores_noise_wc = np.array([matched_filter(y, s)[2] for y in Y_noise_w])

thr_casado = float(np.quantile(scores_noise, 0.99))
thr_wc = float(np.quantile(scores_noise_wc, 0.99))
# raw detection threshold on max|y|
scores_noise_raw = np.array([float(np.max(np.abs(y))) for y in Y_noise])
thr_raw = float(np.quantile(scores_noise_raw, 0.99))
scores_noise_wraw = np.array([float(np.max(np.abs(y))) for y in Y_noise_w])
thr_wraw = float(np.quantile(scores_noise_wraw, 0.99))

fa_casado = float(np.mean(scores_noise >= thr_casado))
fa_wc = float(np.mean(scores_noise_wc >= thr_wc))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_noise, bins=40, alpha=0.75, label="ruído puro")
ax.axvline(thr_casado, color="r", ls="--", label=f"limiar P99={thr_casado:.3f}")
ax.set_xlabel("máx. filtro casado")
ax.set_title("Limiar de detecção (FA ≈ 1%)")
ax.legend()
savefig("07_limiar.png")

# ---------------------------------------------------------------------------
# 7. Full train evaluation
# ---------------------------------------------------------------------------

def eval_set(Y, labels, Y_w_precomputed=None):
    """Evaluate all pipelines on a labeled set."""
    global Y_train_w, Y_test_w  # noqa — use precomputed

    results = {}

    # prepare wiener version
    if Y.shape[0] == Y_train.shape[0]:
        Yw = Y_train_w
    elif Y.shape[0] == Y_test.shape[0]:
        Yw = Y_test_w
    else:
        Yw = np.stack([apply_wiener_freq(y, H_w) for y in Y])

    # "naive" rows form pipelines completos e independentes: deteccao por max|y|,
    # posicao por argmax (corrigida pelo pico do template) e amplitude pelo maximo.
    # "matched" rows: n0 pelo casado e amplitude por BLUE.
    configs = {
        "bruto": (Y, "naive", False),
        "wiener": (Yw, "naive", False),
        "casado": (Y, "matched", False),
        "wiener_casado": (Yw, "matched", False),
        "blue_colorido": (Y, "matched", True),
    }

    y_true = labels.event_present.to_numpy().astype(int)
    peak_offset = int(np.argmax(s))

    for name, (Y_in, det_mode, colored) in configs.items():
        n = len(Y_in)
        scores = np.zeros(n)
        n0h = np.zeros(n, dtype=int)
        Ah = np.zeros(n)
        Yh = np.zeros_like(Y)
        for i in range(n):
            if det_mode == "matched":
                out, n0, sc = matched_filter(Y_in[i], s)
                scores[i] = sc
                n0h[i] = n0
                sal = place_template(s, n0)
                # amplitude sempre estimada na observacao original
                Ah[i] = blue_amplitude(Y[i], sal, Cinv if colored else None)
                Yh[i] = Ah[i] * sal
            else:
                scores[i] = float(np.max(np.abs(Y_in[i])))
                n0h[i] = int(np.clip(np.argmax(Y_in[i]) - peak_offset, 0, N_WIN - N_TMPL))
                Ah[i] = float(np.max(Y_in[i]))
                Yh[i] = Y_in[i]

        if name == "bruto":
            thr = thr_raw
        elif name == "wiener":
            thr = thr_wraw
        elif name == "wiener_casado":
            thr = thr_wc
        else:
            thr = thr_casado

        y_pred = (scores >= thr).astype(int)
        fpr, tpr = roc_curve_scores(y_true, scores)
        auc = auc_trapz(fpr, tpr)
        det = detection_metrics(y_true, y_pred)

        # waveform RMSE: naive rows use Y_in directly (Yh = Y_in); matched rows
        # use the reconstructed pulse A_hat * s[n - n0_hat]
        rmse_wave = rmse_waveform(Yh, labels, s)

        tp_mask = (y_true == 1) & (y_pred == 1)
        mpos = y_true == 1
        n0_abs = np.abs(n0h[mpos] - labels.t0_sample.to_numpy()[mpos])
        results[name] = dict(
            scores=scores,
            n0_hat=n0h,
            A_hat=Ah,
            y_hat=Yh,
            thr=thr,
            y_pred=y_pred,
            auc=auc,
            fpr=fpr,
            tpr=tpr,
            det=det,
            rmse_wave=rmse_wave,
            rmse_amp=rmse_amp(Ah, labels, mask=(y_true == 1)),
            rmse_amp_tp=rmse_amp(Ah, labels, mask=tp_mask),
            n0_err=float(np.mean(n0_abs)) if len(n0_abs) else np.nan,
            n0_med=float(np.median(n0_abs)) if len(n0_abs) else np.nan,
            n0_err_tp=n0_error(n0h, labels, mask=tp_mask),
        )
    return results


train_eval = eval_set(Y_train, lab_train)
test_eval = eval_set(Y_test, lab_test)

# ROC figure (train)
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, label in [
    ("bruto", "Bruto (máx |y|)"),
    ("wiener", "Wiener (máx |y|)"),
    ("casado", "Filtro casado"),
    ("wiener_casado", "Wiener + casado"),
]:
    r = train_eval[name]
    ax.plot(r["fpr"], r["tpr"], label=f"{label} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Taxa de falso positivo")
ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_title("Curvas ROC — treino")
ax.legend(fontsize=8)
savefig("08_roc_treino.png")

# ROC test
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, label in [
    ("bruto", "Bruto"),
    ("wiener", "Wiener"),
    ("casado", "Filtro casado"),
    ("wiener_casado", "Wiener + casado"),
]:
    r = test_eval[name]
    ax.plot(r["fpr"], r["tpr"], label=f"{label} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Taxa de falso positivo")
ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_title("Curvas ROC — teste")
ax.legend(fontsize=8)
savefig("09_roc_teste.png")

# Confusion matrix for best detector on train (casado)
best_det_name = max(["casado", "wiener_casado", "bruto", "wiener"], key=lambda n: train_eval[n]["auc"])
d = train_eval[best_det_name]["det"]
fig, ax = plt.subplots(figsize=(4.5, 4))
cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]], dtype=float)
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, int(v), ha="center", va="center", color="black")
ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
ax.set_yticks([0, 1], ["Real 0", "Real 1"])
ax.set_title(f"Matriz de confusão — {best_det_name} (treino)")
savefig("10_confusao_treino.png")

d_te = test_eval[best_det_name]["det"]
fig, ax = plt.subplots(figsize=(4.5, 4))
cm = np.array([[d_te["tn"], d_te["fp"]], [d_te["fn"], d_te["tp"]]], dtype=float)
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, int(v), ha="center", va="center", color="black")
ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
ax.set_yticks([0, 1], ["Real 0", "Real 1"])
ax.set_title(f"Matriz de confusão — {best_det_name} (teste)")
savefig("11_confusao_teste.png")

# Reconstruction example
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
true_p = true_pulse(s, t0p, Ap)
axes[0].plot(Y_train[idx_pos], lw=0.7, label="y[n]", alpha=0.8)
axes[0].plot(true_p, lw=1.5, label="pulso verdadeiro")
axes[0].legend(fontsize=8)
axes[0].set_title("Observação e pulso verdadeiro")
axes[1].plot(true_p, lw=1.5, label="verdadeiro")
axes[1].plot(train_eval["casado"]["y_hat"][idx_pos], lw=1.2, label="BLUE+casado")
axes[1].plot(Y_train_w[idx_pos], lw=0.8, alpha=0.8, label="Wiener")
axes[1].legend(fontsize=8)
axes[1].set_title("Reconstruções")
axes[1].set_xlabel("n")
savefig("12_reconstrucao_exemplo.png")

# Summary tables
methods_order = ["bruto", "wiener", "casado", "wiener_casado", "blue_colorido"]
labels_pt = {
    "bruto": "Sinal bruto",
    "wiener": "Wiener",
    "casado": "Casado + BLUE (branco)",
    "wiener_casado": "Wiener + casado",
    "blue_colorido": "Casado + BLUE (colorido)",
}


def fmt_pt(x, nd=3):
    """Formata número com vírgula decimal para o LaTeX."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def write_metrics_table(eval_dict, fname):
    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Método & AUC & Prec. & Rev. & F1 & RMSE forma & RMSE $A$ & med$|\Delta n_0|$ & FA emp. \\",
        r"\midrule",
    ]
    for m in methods_order:
        r = eval_dict[m]
        # FA emp on this set: fp / (fp+tn)
        det = r["det"]
        fa = det["fp"] / (det["fp"] + det["tn"]) if (det["fp"] + det["tn"]) else 0
        lines.append(
            f"{labels_pt[m]} & {fmt_pt(r['auc'])} & {fmt_pt(det['precision'])} & "
            f"{fmt_pt(det['recall'])} & {fmt_pt(det['f1'])} & "
            f"{fmt_pt(r['rmse_wave'])} & {fmt_pt(r['rmse_amp'])} & "
            f"{fmt_pt(r['n0_med'], 2)} & {fmt_pt(fa)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIG / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", fname)


write_metrics_table(train_eval, "tab_metricas_treino.tex")
write_metrics_table(test_eval, "tab_metricas_teste.tex")

# Choose best methods per task (train)
best_wave = min(methods_order, key=lambda m: train_eval[m]["rmse_wave"])
best_det = max(["bruto", "wiener", "casado", "wiener_casado"], key=lambda m: train_eval[m]["auc"])
best_amp = min(["casado", "blue_colorido"], key=lambda m: train_eval[m]["rmse_amp"])

# ---------------------------------------------------------------------------
# valores.tex
# ---------------------------------------------------------------------------

def fmt(x, nd=4):
    """Formata float com vírgula decimal (padrão pt-BR) para uso no LaTeX."""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}".replace(".", "{,}")
    return str(x)


vals = {
    "NWin": str(N_WIN),
    "NTmpl": str(N_TMPL),
    "Fs": "1000",
    "NNoise": str(len(Y_noise)),
    "NTrain": str(len(Y_train)),
    "NTest": str(len(Y_test)),
    "MuRuido": fmt(mu_r, 4),
    "VarRuido": fmt(var_r, 4),
    "StdRuido": fmt(std_r, 4),
    "FracAutocorrFora": fmt(outside * 100, 1),
    "PsdRatio": fmt(psd_ratio, 2),
    "NoiseColored": "sim" if noise_is_colored else "não",
    "FPeakPulso": fmt(f_peak, 1),
    "EnergyLowBand": fmt(energy_low * 100, 1),
    "ThrCasado": fmt(thr_casado, 4),
    "ThrWienerCasado": fmt(thr_wc, 4),
    "FaCasado": fmt(fa_casado * 100, 2),
    "NZeroErrMean": fmt(n0_err_mean, 2),
    "NZeroErrMed": fmt(n0_err_med, 2),
    "RmseNaiveA": fmt(rmse_naive, 3),
    "RmseBlueBranco": fmt(rmse_blue_w, 3),
    "RmseBlueColorido": fmt(rmse_blue_c, 3),
    "RmseBrutoWave": fmt(rmse_bruto_wave, 3),
    "RmseWienerWave": fmt(rmse_wiener_wave, 3),
    "BestWave": labels_pt[best_wave],
    "BestDet": labels_pt[best_det],
    "BestAmp": labels_pt[best_amp],
    "AucCasadoTrain": fmt(train_eval["casado"]["auc"], 3),
    "AucCasadoTest": fmt(test_eval["casado"]["auc"], 3),
    "AucWcTrain": fmt(train_eval["wiener_casado"]["auc"], 3),
    "AucWcTest": fmt(test_eval["wiener_casado"]["auc"], 3),
    "FoneCasadoTrain": fmt(train_eval["casado"]["det"]["f1"], 3),
    "FoneCasadoTest": fmt(test_eval["casado"]["det"]["f1"], 3),
    "FoneWcTrain": fmt(train_eval["wiener_casado"]["det"]["f1"], 3),
    "FoneWcTest": fmt(test_eval["wiener_casado"]["det"]["f1"], 3),
    "RmseAmpCasadoTrain": fmt(train_eval["casado"]["rmse_amp"], 3),
    "RmseAmpCasadoTest": fmt(test_eval["casado"]["rmse_amp"], 3),
    "RmseWaveCasadoTrain": fmt(train_eval["casado"]["rmse_wave"], 3),
    "RmseWaveCasadoTest": fmt(test_eval["casado"]["rmse_wave"], 3),
    "TpTrain": str(train_eval[best_det]["det"]["tp"]),
    "FpTrain": str(train_eval[best_det]["det"]["fp"]),
    "FnTrain": str(train_eval[best_det]["det"]["fn"]),
    "TnTrain": str(train_eval[best_det]["det"]["tn"]),
    "TpTest": str(test_eval[best_det]["det"]["tp"]),
    "FpTest": str(test_eval[best_det]["det"]["fp"]),
    "FnTest": str(test_eval[best_det]["det"]["fn"]),
    "TnTest": str(test_eval[best_det]["det"]["tn"]),
    "BestDetKey": best_det.replace("_", ""),
}

lines = ["% Gerado por gerar_figuras.py - nao editar manualmente"]
for k, v in vals.items():
    lines.append(f"\\newcommand{{\\val{k}}}{{{v}}}")
(Path(__file__).resolve().parent / "valores.tex").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print("saved valores.tex")

# print summary
print("\n=== RESUMO TREINO ===")
for m in methods_order:
    r = train_eval[m]
    print(
        f"{m:16s} AUC={r['auc']:.3f} F1={r['det']['f1']:.3f} "
        f"RMSEw={r['rmse_wave']:.3f} RMSEa={r['rmse_amp']:.3f} n0err={r['n0_err']:.2f}"
    )
print("\n=== RESUMO TESTE ===")
for m in methods_order:
    r = test_eval[m]
    print(
        f"{m:16s} AUC={r['auc']:.3f} F1={r['det']['f1']:.3f} "
        f"RMSEw={r['rmse_wave']:.3f} RMSEa={r['rmse_amp']:.3f} n0err={r['n0_err']:.2f}"
    )
print(f"\nMelhor forma: {best_wave}, detecção: {best_det}, amplitude: {best_amp}")
print(f"Ruído colorido? {noise_is_colored} (frac fora={outside:.3f}, psd_ratio={psd_ratio:.2f})")
