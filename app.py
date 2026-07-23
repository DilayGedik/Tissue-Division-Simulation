import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st
from simulation import TissueConfig, simulate_tissue_diffusion, concentration_snapshot_to_dataframe

st.set_page_config(page_title="Tissue Drug Diffusion Simulator", page_icon="🧫", layout="wide")
st.title("Drug Diffusion Through Tissue")
st.caption("Interactive 2D reaction-diffusion model with tumor uptake, degradation, heatmaps, animation, and downloadable results.")

with st.sidebar:
    st.header("Transport")
    diffusion = st.slider("Diffusion coefficient (µm²/s)", 20.0, 500.0, 120.0, 10.0)
    porosity = st.slider("Tissue porosity", 0.20, 1.00, 0.75, 0.05)
    degradation = st.slider("Degradation rate (1/s)", 0.000, 0.020, 0.003, 0.001, format="%.3f")
    st.header("Cellular uptake")
    healthy = st.slider("Healthy uptake (1/s)", 0.000, 0.030, 0.004, 0.001, format="%.3f")
    tumor = st.slider("Tumor uptake (1/s)", 0.000, 0.060, 0.012, 0.002, format="%.3f")
    st.header("Geometry")
    tumor_radius = st.slider("Tumor radius (µm)", 70.0, 240.0, 145.0, 5.0)
    tumor_x = st.slider("Tumor center x (µm)", 450.0, 1050.0, 780.0, 10.0)
    source_radius = st.slider("Injection radius (µm)", 25.0, 120.0, 55.0, 5.0)
    duration = st.slider("Simulation duration (s)", 120.0, 1800.0, 900.0, 60.0)
    run = st.button("Run simulation", type="primary", use_container_width=True)

config = TissueConfig(diffusion_um2_s=diffusion, porosity=porosity,
    degradation_rate_s=degradation, healthy_uptake_rate_s=healthy,
    tumor_uptake_rate_s=tumor, tumor_radius_um=tumor_radius,
    tumor_center_x_um=tumor_x, source_radius_um=source_radius, duration_s=duration)

if run or "result" not in st.session_state:
    with st.spinner("Solving reaction-diffusion model..."):
        st.session_state.result = simulate_tissue_diffusion(config)
        st.session_state.config = config

result = st.session_state.result
active = st.session_state.config
if active != config:
    st.info("Parameters changed. Click Run simulation to update.")

idx = st.slider("Time point", 0, len(result["times_s"])-1, len(result["times_s"])-1)
t = float(result["times_s"][idx])
field = result["concentration"][idx]
metrics = result["metrics"]
tm = float(metrics.loc[idx,"tumor_mean_concentration"])
hm = float(metrics.loc[idx,"healthy_mean_concentration"])
pd = float(metrics.loc[idx,"penetration_depth_um"])

a,b,c,d = st.columns(4)
a.metric("Current time", f"{t:.0f} s")
b.metric("Tumor concentration", f"{tm:.3f}")
c.metric("Penetration depth", f"{pd:.0f} µm")
d.metric("Tumor / healthy ratio", f"{tm/max(hm,1e-12):.2f}")

tabs = st.tabs(["Concentration map","Time series","Absorption","Export & assumptions"])

with tabs[0]:
    fig, ax = plt.subplots(figsize=(11,6))
    im = ax.imshow(field, origin="lower", extent=[result["x_um"][0],result["x_um"][-1],
        result["y_um"][0],result["y_um"][-1]], aspect="auto", vmin=0, vmax=1)
    ax.contour(result["x_um"],result["y_um"],result["tumor_mask"].astype(float),levels=[0.5],linewidths=2)
    ax.contour(result["x_um"],result["y_um"],result["source_mask"].astype(float),levels=[0.5],linewidths=2,linestyles="--")
    ax.set(title=f"Drug concentration at t = {t:.0f} s", xlabel="x (µm)", ylabel="y (µm)")
    fig.colorbar(im, ax=ax, label="Normalized concentration")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Solid outline: tumor. Dashed outline: injection site.")

    if st.button("Generate animated GIF"):
        with st.spinner("Rendering animation..."):
            fig, ax = plt.subplots(figsize=(9,5))
            image = ax.imshow(result["concentration"][0], origin="lower",
                extent=[result["x_um"][0],result["x_um"][-1],result["y_um"][0],result["y_um"][-1]],
                aspect="auto", vmin=0, vmax=1)
            ax.contour(result["x_um"],result["y_um"],result["tumor_mask"].astype(float),levels=[0.5])
            def update(i):
                image.set_data(result["concentration"][i])
                ax.set_title(f"Drug diffusion: t = {result['times_s'][i]:.0f} s")
                return [image]
            ani = FuncAnimation(fig, update, frames=len(result["times_s"]), interval=80)
            path = "tissue_diffusion_animation.gif"
            ani.save(path, writer=PillowWriter(fps=12))
            plt.close(fig)
            data = open(path,"rb").read()
            st.image(data)
            st.download_button("Download GIF", data=data, file_name=path, mime="image/gif")

with tabs[1]:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(metrics["time_s"],metrics["tumor_mean_concentration"],label="Tumor")
    ax.plot(metrics["time_s"],metrics["healthy_mean_concentration"],label="Healthy tissue")
    ax.set(title="Average concentration over time",xlabel="Time (s)",ylabel="Normalized concentration")
    ax.grid(True,alpha=.25); ax.legend()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(metrics["time_s"],metrics["penetration_depth_um"])
    ax.set(title="Drug penetration depth",xlabel="Time (s)",ylabel="Depth (µm)")
    ax.grid(True,alpha=.25)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

with tabs[2]:
    st.line_chart(metrics.set_index("time_s")[["tumor_absorbed_integral","healthy_absorbed_integral"]])
    ta = float(metrics["tumor_absorbed_integral"].iloc[-1])
    ha = float(metrics["healthy_absorbed_integral"].iloc[-1])
    st.metric("Tumor share of absorbed drug", f"{100*ta/max(ta+ha,1e-12):.1f}%")

with tabs[3]:
    st.dataframe(metrics,use_container_width=True)
    st.download_button("Download metrics CSV",metrics.to_csv(index=False).encode(),
        "tissue_diffusion_metrics.csv","text/csv")
    snap = concentration_snapshot_to_dataframe(result,idx)
    st.download_button("Download selected concentration map",snap.to_csv(index=False).encode(),
        f"concentration_map_{t:.0f}s.csv","text/csv")
    st.markdown("""### Assumptions
- Homogeneous 2D porous tissue
- First-order degradation and uptake
- Different healthy and tumor uptake rates
- Fixed-concentration injection region
- Zero-flux outer boundaries
- No perfusion, nonlinear binding, or cell mechanics

This is a reduced-order portfolio and early-design model, not a clinical prediction tool.""")
