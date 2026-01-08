# launcher.py
# Petit launcher GUI: lance tes scripts existants + comparateur avant/après
# NE MODIFIE AUCUN FICHIER EXISTANT.

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Matplotlib (affichage comparateur)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import cv2 as cv
except Exception:
    cv = None


PY = sys.executable

PHASE1_SCRIPT = "./erosion.py"
PHASE2_SCRIPT = "./phase2_upgraded.py"

# ---------------------------------------------------------------


def _read_img(path: str) -> np.ndarray:
    """Lit PNG/JPG/TIF etc + FITS (si astropy installé). Retour float32."""
    ext = os.path.splitext(path)[1].lower()

    # images classiques
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        if cv is not None:
            im = cv.imread(path, cv.IMREAD_UNCHANGED)
            if im is None:
                raise ValueError("Image illisible.")
            if im.ndim == 3:
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            im = im.astype(np.float32)
            if im.max() > 1:
                im /= 255.0
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3]
            return im
        else:
            import matplotlib.image as mpimg
            im = mpimg.imread(path).astype(np.float32)
            if im.max() > 1:
                im /= 255.0
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3]
            return im

    # FITS
    if ext in [".fits", ".fit", ".fts"]:
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data
        if data is None:
            raise ValueError("FITS vide ou invalide.")
        data = np.array(data, dtype=np.float32)
        # si 2D => ok ; si 3D => essaye HWC ou CHW
        if data.ndim == 3:
            # si (C,H,W)
            if data.shape[0] in (1, 3) and data.shape[1] > 10 and data.shape[2] > 10:
                data = np.transpose(data, (1, 2, 0))
            # si (H,W,C) => ok
        return data

    raise ValueError("Format non supporté. Utilise FITS/PNG/JPG.")


def _to01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn > 1e-12:
        x = (x - mn) / (mx - mn)
    return np.clip(x, 0, 1)


def _split_view(before: np.ndarray, after: np.ndarray, ratio: float, after_left: bool) -> np.ndarray:
    b = _to01(before)
    a = _to01(after)

    # harmonise tailles (crop centre)
    h = min(b.shape[0], a.shape[0])
    w = min(b.shape[1], a.shape[1])

    def crop(x):
        y0 = (x.shape[0] - h) // 2
        x0 = (x.shape[1] - w) // 2
        return x[y0:y0+h, x0:x0+w] if x.ndim == 2 else x[y0:y0+h, x0:x0+w, :]

    b, a = crop(b), crop(a)

    # grayscale -> rgb
    if b.ndim == 2:
        b = np.repeat(b[:, :, None], 3, axis=2)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)

    # remove alpha
    if b.shape[2] == 4:
        b = b[:, :, :3]
    if a.shape[2] == 4:
        a = a[:, :, :3]

    s = int(np.clip(ratio, 0, 1) * w)
    out = b.copy()
    if after_left:
        out[:, :s, :] = a[:, :s, :]
    else:
        out[:, s:, :] = a[:, s:, :]
    return out


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STAR REDUCTION — Launcher (Phase 1/2 + Comparateur)")
        self.geometry("1100x680")

        self.fits_path = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value=os.path.abspath("results"))

        self.before_path = tk.StringVar(value="")
        self.after_path = tk.StringVar(value="")
        self.after_left = tk.BooleanVar(value=True)
        self.ratio = tk.DoubleVar(value=0.5)

        self.before_img = None
        self.after_img = None

        self.result_fig = Figure(figsize=(7.2, 4.3), dpi=110)
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_axis_off()
        self.result_canvas = None  # will be set in build

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Fichier FITS (entrée):").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.fits_path, width=75).grid(row=0, column=1, padx=8, sticky="we")
        ttk.Button(top, text="Parcourir", command=self.pick_fits).grid(row=0, column=2)

        ttk.Label(top, text="Dossier sortie:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.out_dir, width=75).grid(row=1, column=1, padx=8, pady=(8, 0), sticky="we")
        ttk.Button(top, text="Choisir", command=self.pick_out).grid(row=1, column=2, pady=(8, 0))

        top.columnconfigure(1, weight=1)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(fill=tk.X)

        ttk.Button(btns, text="▶ Phase 1 (erosion.py)", command=self.run_phase1).pack(side=tk.LEFT)
        ttk.Button(btns, text="▶ Phase 2 (phase2_upgraded.py)", command=self.run_phase2).pack(side=tk.LEFT, padx=8)

        ttk.Separator(self).pack(fill=tk.X, padx=10, pady=6)

        # Résultats
        res = ttk.LabelFrame(self, text="Résultats de la phase", padding=10)
        res.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=res)
        self.result_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Separator(self).pack(fill=tk.X, padx=10, pady=6)

        # Comparateur
        comp = ttk.LabelFrame(self, text="Comparateur (Avant/Après)", padding=10)
        comp.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        row = ttk.Frame(comp)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Avant:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.before_path, width=65).grid(row=0, column=1, padx=8, sticky="we")
        ttk.Button(row, text="Charger", command=self.pick_before).grid(row=0, column=2)

        ttk.Label(row, text="Après:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(row, textvariable=self.after_path, width=65).grid(row=1, column=1, padx=8, pady=(8, 0), sticky="we")
        ttk.Button(row, text="Charger", command=self.pick_after).grid(row=1, column=2, pady=(8, 0))

        row.columnconfigure(1, weight=1)

        opt = ttk.Frame(comp)
        opt.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(opt, text="Après à gauche", variable=self.after_left, command=self.render_compare).pack(side=tk.LEFT)
        ttk.Label(opt, text="Split:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Scale(opt, from_=0.0, to=1.0, variable=self.ratio,
                  command=lambda _=None: self.render_compare()).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Canvas
        self.fig = Figure(figsize=(7.2, 4.3), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=comp)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log
        self.log = tk.Text(self, height=8)
        self.log.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))
        self._log("Prêt.\n")

    def _log(self, s: str):
        self.log.insert("end", s + ("\n" if not s.endswith("\n") else ""))
        self.log.see("end")

    def pick_fits(self):
        p = filedialog.askopenfilename(
            title="Choisir un FITS",
            filetypes=[("FITS", "*.fits *.fit *.fts"), ("Tous", "*.*")]
        )
        if p:
            self.fits_path.set(p)

    def pick_out(self):
        d = filedialog.askdirectory(title="Choisir dossier sortie")
        if d:
            self.out_dir.set(d)

    def _ensure_fits_and_out(self):
        fp = self.fits_path.get().strip()
        od = self.out_dir.get().strip()
        if not fp or not os.path.exists(fp):
            raise ValueError("Sélectionne un FITS valide.")
        os.makedirs(od, exist_ok=True)
        return fp, od

    def _run_script(self, script_name: str, args: list[str]):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror("Erreur", f"Fichier introuvable: {script_path}")
            return

        cmd = [PY, script_path] + args
        self._log("CMD: " + " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            if p.stdout.strip():
                self._log(p.stdout.strip())
            if p.stderr.strip():
                self._log(p.stderr.strip())
            if p.returncode != 0:
                raise RuntimeError(f"Le script a échoué (code {p.returncode}).")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # --- Boutons phases (sans modifier tes fichiers) ---
    def run_phase1(self):
        try:
            fp, od = self._ensure_fits_and_out()
            # Beaucoup de scripts acceptent: input output ; si le tien est différent, dis-moi et je te change 1 ligne.
            self._run_script(PHASE1_SCRIPT, [fp, od])
            self.display_result(os.path.join(od, 'comparaison_phase1.png'))
        except Exception as e:
            messagebox.showerror("Phase 1", str(e))

    def run_phase2(self):
        try:
            fp, od = self._ensure_fits_and_out()
            self._run_script(PHASE2_SCRIPT, [fp, od])
            self.display_result(os.path.join(od, 'comparaison_phase2.png'))
        except Exception as e:
            messagebox.showerror("Phase 2", str(e))

    # --- Comparateur ---
    def pick_before(self):
        p = filedialog.askopenfilename(
            title="Charger AVANT",
            filetypes=[("Images/FITS", "*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not p:
            return
        self.before_path.set(p)
        try:
            self.before_img = _read_img(p)
            self._log(f"Avant chargé: {p}")
            self.render_compare()
        except Exception as e:
            messagebox.showerror("Avant", str(e))

    def pick_after(self):
        p = filedialog.askopenfilename(
            title="Charger APRÈS",
            filetypes=[("Images/FITS", "*.fits *.fit *.fts *.png *.jpg *.jpeg *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not p:
            return
        self.after_path.set(p)
        try:
            self.after_img = _read_img(p)
            self._log(f"Après chargé: {p}")
            self.render_compare()
        except Exception as e:
            messagebox.showerror("Après", str(e))

    def render_compare(self):
        if self.before_img is None or self.after_img is None:
            return
        img = _split_view(
            self.before_img, self.after_img,
            ratio=float(self.ratio.get()),
            after_left=bool(self.after_left.get())
        )
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(img, interpolation="nearest")
        self.canvas.draw_idle()

    def display_result(self, path: str):
        try:
            img = _read_img(path)
            self.result_ax.clear()
            self.result_ax.set_axis_off()
            self.result_ax.imshow(img, interpolation="nearest")
            self.result_canvas.draw_idle()
            self._log(f"Résultat affiché: {path}")
        except Exception as e:
            self._log(f"Erreur affichage résultat: {e}")


if __name__ == "__main__":
    Launcher().mainloop()
