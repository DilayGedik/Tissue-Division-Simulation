from simulation import TissueConfig, simulate_tissue_diffusion

config = TissueConfig(diffusion_um2_s=120, duration_s=900)
result = simulate_tissue_diffusion(config)
result["metrics"].to_csv("tissue_diffusion_metrics.csv", index=False)
print(result["metrics"].tail())
