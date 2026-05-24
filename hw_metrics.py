"""
hw_metrics.py — Hardware CSV classification metrics using Valsalva as abnormal proxy.
Normal  = sessions 0,1,2 (supine, head-up, head-down)
Abnormal proxy = session 3 (valsalva+recovery) — known ICP elevation
"""
from __future__ import annotations
import pickle
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import pywt
import xgboost as xgb
from scipy import signal as sp_signal
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CSVS = [
    Path("C:/Users/asus/Downloads/icp_1_27min.csv"),
    Path("C:/Users/asus/Downloads/icp_2.csv"),
]
MODEL_QT = Path("models/xgb_qt.json")
QT_PKL   = Path("models/qt_scaler.pkl")
OUT_DIR  = Path("results/qt_pipeline")

FS=50; WIN=500; STEP=250
FEATURES = ["cardiac_amplitude","cardiac_frequency","respiratory_amplitude",
            "slow_wave_power","cardiac_power"]
SESSION_NAMES = {0:"Supine",1:"Head-Up 30°",2:"Head-Down 10°",3:"Valsalva"}

BLUE="#1565C0"; RED="#C62828"; GREEN="#2E7D32"
ORANGE="#E65100"; GRAY="#546E7A"; LGRAY="#ECEFF1"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,
    "font.family":"DejaVu Sans","font.size":11,
    "axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.7,
})

_nyq = FS/2.0
_B_CARD,_A_CARD = sp_signal.butter(4,[1.0/_nyq,2.5/_nyq],btype="band")
_B_RESP,_A_RESP = sp_signal.butter(4,[0.1/_nyq,0.5/_nyq],btype="band")
_FREQS = np.fft.rfftfreq(WIN,d=1.0/FS)
_FREQ_MASK = (_FREQS>=0.7)&(_FREQS<=2.5)


def extract_window(ir_raw,disp_raw):
    ir=ir_raw.astype(np.float64); disp=disp_raw.astype(np.float64)
    if ir.std()<5.0 or disp.std()<0.05: return None
    ir_dt=sp_signal.detrend(ir); disp_dt=sp_signal.detrend(disp)
    c=sp_signal.filtfilt(_B_CARD,_A_CARD,ir_dt)
    card_amp=float(np.percentile(c,99)-np.percentile(c,1))
    pwr=np.abs(np.fft.rfft(ir_dt))**2
    if not _FREQ_MASK.any(): return None
    card_freq=float(_FREQS[_FREQ_MASK][np.argmax(pwr[_FREQ_MASK])])
    r=sp_signal.filtfilt(_B_RESP,_A_RESP,disp_dt)
    resp_amp=float(np.percentile(r,99)-np.percentile(r,1))
    coeffs=pywt.wavedec(disp_dt,"db4",level=5)
    energies=[float(np.sum(cc**2)) for cc in coeffs]
    total=sum(energies)+1e-12
    slow_pow=energies[0]/total; cardiac_pow=energies[2]/total
    feat=np.array([card_amp,card_freq,resp_amp,slow_pow,cardiac_pow],dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def load_csv(path):
    df=pd.read_csv(path,comment="#")
    df=df[df["artifact_flag"]==0].reset_index(drop=True)
    feats,sessions=[],[]
    n_win=(len(df)-WIN)//STEP+1
    for w in range(n_win):
        s,e=w*STEP,w*STEP+WIN
        sl=df.iloc[s:e]
        feat=extract_window(sl["ir_raw"].values,sl["disp_raw"].values)
        if feat is None: continue
        feats.append(feat)
        sessions.append(int(sl["session_label"].mode()[0]))
    return np.array(feats,dtype=np.float32), np.array(sessions,dtype=np.int32)


def compute_metrics(y_true, probs, thr):
    preds=(probs>=thr).astype(int)
    cm=confusion_matrix(y_true,preds)
    tn,fp,fn,tp=cm.ravel()
    return {
        "auc":          round(float(roc_auc_score(y_true,probs)),4),
        "avg_precision":round(float(average_precision_score(y_true,probs)),4),
        "f1":           round(float(f1_score(y_true,preds)),4),
        "precision":    round(float(precision_score(y_true,preds,zero_division=0)),4),
        "recall":       round(float(recall_score(y_true,preds)),4),
        "specificity":  round(float(tn/(tn+fp)),4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true,preds)),4),
        "threshold":    round(float(thr),4),
        "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
    }


def youden_threshold(y_true, probs):
    fpr,tpr,thresholds=roc_curve(y_true,probs)
    j=tpr-fpr
    return float(thresholds[np.argmax(j)])


def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)

    # Load model + QT
    bst=xgb.Booster(); bst.load_model(str(MODEL_QT))
    with open(QT_PKL,"rb") as f: qt=pickle.load(f)
    print("Model and QT scaler loaded.\n")

    all_X,all_sess,all_file=[],[],[]
    for csv in CSVS:
        X,sess=load_csv(csv)
        all_X.append(X); all_sess.append(sess)
        all_file.extend([csv.name]*len(X))
        print(f"{csv.name}: {len(X)} windows")

    X_all   = np.vstack(all_X)
    sess_all= np.concatenate(all_sess)

    # QT align
    X_qt = qt.transform(X_all)
    dm   = xgb.DMatrix(X_qt, feature_names=FEATURES)
    probs= bst.predict(dm)

    # Pseudo labels: 0=normal sessions (0,1,2), 1=Valsalva (3)
    y_pseudo = (sess_all==3).astype(int)

    print(f"\nPseudo-label distribution:")
    print(f"  Normal (sessions 0-2): {(y_pseudo==0).sum()} windows")
    print(f"  Abnormal proxy (Valsalva): {(y_pseudo==1).sum()} windows")

    # Best threshold on this data (Youden's J)
    thr_youden = youden_threshold(y_pseudo, probs)
    thr_train  = 0.2953  # from CHARIS training

    print(f"\nUsing CHARIS threshold : {thr_train}")
    print(f"Youden J on HW data   : {thr_youden:.4f}")

    m_train = compute_metrics(y_pseudo, probs, thr_train)
    m_youden= compute_metrics(y_pseudo, probs, thr_youden)

    SEP="-"*55
    for label,m in [("CHARIS threshold (0.2953)",m_train),
                    ("HW-optimised threshold (Youden J)",m_youden)]:
        print(f"\n{'='*55}")
        print(f"  [{label}]")
        print(SEP)
        print(f"  AUC-ROC          : {m['auc']}")
        print(f"  Avg Precision(AP): {m['avg_precision']}")
        print(f"  F1-Score         : {m['f1']}")
        print(f"  Precision        : {m['precision']}")
        print(f"  Recall/Sensitivity:{m['recall']}")
        print(f"  Specificity      : {m['specificity']}")
        print(f"  Balanced Accuracy: {m['balanced_acc']}")
        print(f"  Threshold        : {m['threshold']}")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18,12))
    fig.suptitle("Hardware CSV — Classification Metrics\n"
                 "(Normal=Supine/Tilt sessions  |  Abnormal proxy=Valsalva windows)",
                 fontsize=13, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.38)

    m = m_train  # use CHARIS threshold for all plots

    # ── 1. Metrics scorecard ─────────────────────────────────────────────────
    ax0=fig.add_subplot(gs[0,0]); ax0.axis("off")
    rows=[
        ("AUC-ROC",        f"{m['auc']:.4f}",         "> 0.80", GREEN if m['auc']>0.80 else ORANGE),
        ("Avg Precision",  f"{m['avg_precision']:.4f}","> 0.60", GREEN if m['avg_precision']>0.60 else ORANGE),
        ("F1-Score",       f"{m['f1']:.4f}",           "> 0.60", GREEN if m['f1']>0.60 else ORANGE),
        ("Precision",      f"{m['precision']:.4f}",    "—",      GRAY),
        ("Recall (Sens.)", f"{m['recall']:.4f}",       "> 0.70", GREEN if m['recall']>0.70 else RED),
        ("Specificity",    f"{m['specificity']:.4f}",  "> 0.90", GREEN if m['specificity']>0.90 else ORANGE),
        ("Balanced Acc.",  f"{m['balanced_acc']:.4f}", "> 0.75", GREEN if m['balanced_acc']>0.75 else ORANGE),
        ("Threshold",      f"{m['threshold']:.4f}",    "CHARIS J",GRAY),
    ]
    y0=0.97
    ax0.text(0.0,y0+0.04,"Hardware Metrics Scorecard",fontsize=11,fontweight="bold",
             transform=ax0.transAxes,color=BLUE)
    ax0.text(0.0,y0-0.03,"(Valsalva = abnormal proxy)",fontsize=8.5,
             transform=ax0.transAxes,color=GRAY,style="italic")
    for i,(label,val,tgt,col) in enumerate(rows):
        yp=y0-0.02-i*0.108
        bg=LGRAY if i%2==0 else "white"
        ax0.add_patch(FancyBboxPatch((0,yp-0.045),1.0,0.09,
                       transform=ax0.transAxes,boxstyle="square,pad=0",
                       facecolor=bg,edgecolor="none",zorder=0))
        ax0.text(0.02,yp,label,transform=ax0.transAxes,fontsize=9,va="center")
        ax0.text(0.60,yp,val,  transform=ax0.transAxes,fontsize=10,va="center",
                 fontweight="bold",color=col)
        ax0.text(0.82,yp,tgt,  transform=ax0.transAxes,fontsize=8,va="center",color=GRAY)
    ax0.set_xlim(0,1); ax0.set_ylim(-0.1,1.08)

    # ── 2. ROC Curve ─────────────────────────────────────────────────────────
    ax1=fig.add_subplot(gs[0,1])
    fpr,tpr,_=roc_curve(y_pseudo,probs)
    ax1.plot(fpr,tpr,color=BLUE,lw=2,label=f"AUC = {m['auc']:.4f}")
    ax1.plot([0,1],[0,1],"--",color=GRAY,lw=1,label="Random")
    ax1.scatter([1-m['specificity']],[m['recall']],color=RED,s=100,zorder=5,
                label=f"Op. point (thr={m['threshold']:.3f})")
    ax1.set(xlabel="False Positive Rate",ylabel="True Positive Rate",
            title="ROC Curve — Hardware\n(Valsalva as abnormal proxy)")
    ax1.legend(fontsize=9); ax1.set_xlim(-0.02,1.02); ax1.set_ylim(-0.02,1.02)

    # ── 3. Precision-Recall Curve ─────────────────────────────────────────────
    ax2=fig.add_subplot(gs[0,2])
    prec_c,rec_c,_=precision_recall_curve(y_pseudo,probs)
    baseline=y_pseudo.mean()
    ax2.plot(rec_c,prec_c,color=ORANGE,lw=2,label=f"AP = {m['avg_precision']:.4f}")
    ax2.axhline(baseline,color=GRAY,ls="--",lw=1,label=f"Baseline ({baseline:.2f})")
    ax2.scatter([m['recall']],[m['precision']],color=RED,s=100,zorder=5)
    ax2.set(xlabel="Recall",ylabel="Precision",
            title="Precision-Recall Curve — Hardware")
    ax2.legend(fontsize=9); ax2.set_xlim(-0.02,1.02); ax2.set_ylim(-0.02,1.08)

    # ── 4. Confusion Matrix ───────────────────────────────────────────────────
    ax3=fig.add_subplot(gs[1,0])
    cm_arr=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    total=cm_arr.sum()
    im=ax3.imshow(cm_arr,cmap="Blues",aspect="auto")
    for i in range(2):
        for j in range(2):
            val=cm_arr[i,j]
            ax3.text(j,i,f"{val:,}\n({100*val/total:.1f}%)",
                     ha="center",va="center",fontsize=11,fontweight="bold",
                     color="white" if val>cm_arr.max()*0.6 else "black")
    ax3.set(xticks=[0,1],yticks=[0,1],
            xticklabels=["Pred Normal","Pred Abnormal"],
            yticklabels=["True Normal","True Abnormal"],
            title=f"Confusion Matrix (thr={m['threshold']:.4f})")
    plt.colorbar(im,ax=ax3,fraction=0.046,pad=0.04)

    # ── 5. Score distribution per session ────────────────────────────────────
    ax4=fig.add_subplot(gs[1,1])
    colors_s=[GREEN,BLUE,ORANGE,RED]
    for si in sorted(np.unique(sess_all)):
        mask=sess_all==si
        ax4.hist(probs[mask],bins=40,alpha=0.6,color=colors_s[si],
                 label=f"{SESSION_NAMES[si]} (n={mask.sum()})",density=True)
    ax4.axvline(thr_train,color="black",ls="--",lw=1.5,label=f"Thr={thr_train}")
    ax4.set(xlabel="P(Abnormal)",ylabel="Density",
            title="Score Distribution by Session")
    ax4.legend(fontsize=8.5)

    # ── 6. Per-file metrics comparison ───────────────────────────────────────
    ax5=fig.add_subplot(gs[1,2])
    file_metrics=[]
    for csv,X_hw,sess_hw in zip(CSVS,all_X,all_sess):
        X_qt_f=qt.transform(X_hw)
        dm_f=xgb.DMatrix(X_qt_f,feature_names=FEATURES)
        prob_f=bst.predict(dm_f)
        y_f=(sess_hw==3).astype(int)
        if y_f.sum()==0 or y_f.sum()==len(y_f):
            file_metrics.append({"file":csv.name,"auc":np.nan,"f1":np.nan,
                                  "recall":np.nan,"spec":np.nan})
            continue
        mf=compute_metrics(y_f,prob_f,thr_train)
        file_metrics.append({"file":csv.name,"auc":mf["auc"],"f1":mf["f1"],
                              "recall":mf["recall"],"spec":mf["specificity"]})

    metric_names=["AUC","F1","Recall","Specificity"]
    x=np.arange(len(metric_names)); w=0.35
    for fi,fm in enumerate(file_metrics):
        vals=[fm["auc"],fm["f1"],fm["recall"],fm["spec"]]
        bars=ax5.bar(x+(fi-0.5)*w,vals,w,
                     label=fm["file"].replace(".csv",""),
                     color=BLUE if fi==0 else GREEN,alpha=0.8)
        for bar,v in zip(bars,vals):
            ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                     f"{v:.3f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax5.axhline(1.0,color=GRAY,ls=":",lw=0.8)
    ax5.set(xticks=x,xticklabels=metric_names,ylim=[0,1.15],
            title="Per-File Metric Comparison",ylabel="Score")
    ax5.legend(fontsize=9)

    plt.savefig(OUT_DIR/"fig4_hw_metrics.png",dpi=150,bbox_inches="tight")
    print(f"\n  Saved: fig4_hw_metrics.png")

    # Print combined table
    print(f"\n{'='*55}")
    print(f"  PER-FILE METRICS (CHARIS threshold={thr_train})")
    print(f"{'='*55}")
    print(f"  {'File':<22} {'AUC':>7} {'F1':>7} {'Recall':>8} {'Spec':>8}")
    print(f"  {'-'*52}")
    for fm in file_metrics:
        print(f"  {fm['file']:<22} {fm['auc']:>7.4f} {fm['f1']:>7.4f} "
              f"{fm['recall']:>8.4f} {fm['spec']:>8.4f}")
    print(f"\n  Combined (both files):")
    print(f"  AUC      : {m_train['auc']}")
    print(f"  F1       : {m_train['f1']}")
    print(f"  Recall   : {m_train['recall']}")
    print(f"  Precision: {m_train['precision']}")
    print(f"  Spec.    : {m_train['specificity']}")
    print(f"  Bal.Acc  : {m_train['balanced_acc']}")
    print(f"  Avg Prec : {m_train['avg_precision']}")


if __name__=="__main__":
    main()
