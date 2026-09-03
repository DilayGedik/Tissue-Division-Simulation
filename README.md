#Tissue Drug Diffusion Simulator

Interactive 2D reaction diffusion model for drug transport through tissue.

##Visual outputs
- Time controlled concentration heatmap
- Tumor and injection region outlines
- Downloadable animated GIF
- Tumor and healthy concentration curves
- Penetration depth graph
- Tumor versus healthy absorption graph

##Model
The solver uses:

    dC/dt = D_eff ∇²C - (k_degradation + k_uptake) C

It uses an explicit finite difference method, a stability limited time step,
zero flux outer boundaries, and distinct healthy/tumor uptake regions.

##Run
    python -m venv .venv
    .venv\Scripts\activate       # Windows
    source .venv/bin/activate      # macOS/Linux
    pip install -r requirements.txt
    streamlit run app.py

##Scope
This is a reduced order educational and early design model. It is not a
validated clinical or pharmacokinetic prediction tool.
