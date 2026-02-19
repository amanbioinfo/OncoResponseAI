import shap
import matplotlib.pyplot as plt

def shap_explain(model, X):
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    return fig



def shap_single(model, X_row):
    booster = model.get_booster()   # 👈 critical
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_row)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    return fig

