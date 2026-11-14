# LACTU2030 - Partie statique : modélisation de la mortalité

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

pd.set_option("display.precision", 8)
pd.set_option("display.width", 200)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Définition des dossiers de travail
DOSSIER = Path(__file__).resolve().parent
RESULTATS = DOSSIER / "resultats"
RESULTATS.mkdir(exist_ok=True)

# Fonction pour sauvegarder les figures
def sauvegarder_figure(nom):
    plt.tight_layout()
    plt.savefig(RESULTATS / nom, dpi=150)
    plt.close()


# Q1 - Modélisation par minimisation de l'erreur quadratique

# Chargement et nettoyage des données issues du Human Mortality Database (HMD)

deces = pd.read_csv(DOSSIER / "data" / "Deaths_1x1.txt", sep=r"\s+", skiprows=2, na_values=".")
expo = pd.read_csv(DOSSIER / "data" / "Exposures_1x1.txt", sep=r"\s+", skiprows=2, na_values=".")

# Les âges sont lus au format texte (ex: "110+") : on supprime le "+" et on convertit en nombre
deces["Age"] = deces["Age"].astype(str).str.replace("+", "", regex=False).astype(float)
expo["Age"] = expo["Age"].astype(str).str.replace("+", "", regex=False).astype(float)


# Taux bruts : fonction qui extrait une année et un sexe
def donnees(annee, sexe):
    d = deces[(deces.Year == annee) & (deces.Age >= 30) & (deces.Age <= 105)]
    e = expo[(expo.Year == annee) & (expo.Age >= 30) & (expo.Age <= 105)]
    tab = pd.DataFrame({
        "age": d.Age.values,
        "mu": d[sexe].values / e[sexe].values,  # taux brut = décès / exposition
        "expo": e[sexe].values,
    })
    # Vérifications de cohérence : exactement 76 âges (30 à 105), âges alignés et taux strictement positifs
    assert len(tab) == 76 and (d.Age.values == e.Age.values).all() and (tab.mu > 0).all()
    return tab

# Chargement des 4 jeux de données d'analyse
f2020 = donnees(2020, "Female")
f2023 = donnees(2023, "Female")
h2020 = donnees(2020, "Male")
h2023 = donnees(2023, "Male")
jeux = {"F2020": f2020, "F2023": f2023, "H2020": h2020, "H2023": h2023}


# Le modèle paramétrique
def mu_modele(parametres, age):
    a, b, c = parametres
    return np.exp(a + b * age + c * (age - 70) * (age >= 70))


# L'erreur à minimiser (MSE) 
def mse(parametres, age, mu):
    return np.mean((mu - mu_modele(parametres, age)) ** 2)
 
 
# Minimisation de la MSE 

# Régression linéaire de log(mu) pour initialiser l'optimiseur
def depart(donnees):   
    X = np.column_stack([np.ones(len(donnees)), donnees.age, np.maximum(donnees.age - 70, 0)])
    return np.linalg.lstsq(X, np.log(donnees.mu), rcond=None)[0]


def ajuster_mse(donnees):
    res = minimize(
        mse, depart(donnees), args=(donnees.age.values, donnees.mu.values),
        method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000}
    )
    assert res.success
    return res.x


tableau_mse = pd.DataFrame({nom: ajuster_mse(tab) for nom, tab in jeux.items()},
                                 index=["a", "b", "c"]).T

print("\nParamètres optimisés par MSE")
print(tableau_mse)
tableau_mse.to_csv(RESULTATS / "parametres_mse.csv")


# Espérance de vie (modèle MSE) 
def esperance(parametres, x, age_max=105):
  mu = mu_modele(parametres, np.arange(x, age_max))
  survie_fin = np.exp(-np.cumsum(mu))
  survie_debut = np.concatenate(([1.0], survie_fin[:-1]))
  return np.sum(survie_debut * (-np.expm1(-mu)) / mu)

ages = [40, 50, 60, 70]
esperances_mse = pd.DataFrame(
    {nom: [esperance(tableau_mse.loc[nom], x) for x in ages] for nom in jeux},
    index=[f"e{x}" for x in ages]).T.round(2)


# Q2 - Modélisation par maximisation de la log-vraisemblance

def negative_log_likelihood(theta, age, mu, expo):
    mu_theo = mu_modele(theta, age)
    sigma_carre = mu_theo / expo  # Var(mu_hat) = mu / E_x (décès ~ Poisson(mu * E_x))
    return 0.5 * np.sum(np.log(2 * np.pi * sigma_carre) + (mu - mu_theo) ** 2 / sigma_carre)

def ajuster_log_like(donnees):
    res = minimize(negative_log_likelihood, depart(donnees),
                   args=(donnees.age.values, donnees.mu.values, donnees.expo.values),
                   method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    assert res.success
    return res.x

# Tableau des paramètres ajustés par log-vraisemblance
tableau_log_like = pd.DataFrame({nom: ajuster_log_like(tab) for nom, tab in jeux.items()},
                                index=["a", "b", "c"]).T
print("\nParamètres optimisés par log-vraisemblance")
print(tableau_log_like)
tableau_log_like.to_csv(RESULTATS / "parametres_log_vraisemblance.csv")


# Comparaison des MSE obtenues selon la méthode d'optimisation
comparaison_mse = pd.DataFrame({
    "MSE": [mse(tableau_mse.loc[nom], tab.age, tab.mu) for nom, tab in jeux.items()],
    "MSE_via_likelihood": [mse(tableau_log_like.loc[nom], tab.age, tab.mu) for nom, tab in jeux.items()],
}, index=jeux.keys())

print("\nComparaison des valeurs de MSE")
print(comparaison_mse)
comparaison_mse.to_csv(RESULTATS / "comparaison_mse_vs_likelihood.csv")


# Comparaison des espérances de vie et calcul des biais 
esperances_log_like = pd.DataFrame(
    {nom: [esperance(tableau_log_like.loc[nom], x) for x in ages] for nom in jeux},
    index=[f"e{x}" for x in ages]).T.round(2)

print("\nEspérances de vie estimées (Modèle MSE)")
print(esperances_mse)
esperances_mse.to_csv(RESULTATS / "esperances_vie_mse.csv")

print("\nEspérances de vie estimées (Modèle Log-Vraisemblance)")
print(esperances_log_like)
esperances_log_like.to_csv(RESULTATS / "esperances_vie_log_like.csv")

print("\nÉcart (Log-Vraisemblance - MSE) sur les espérances de vie")
print(esperances_log_like - esperances_mse)
(esperances_log_like - esperances_mse).to_csv(RESULTATS / "ecart_esperances_loglike_mse.csv")

def esperance_brute(tab, x, age_max=105):
  mu = tab.mu[(tab.age >= x) & (tab.age <= age_max - 1)].values
  survie_fin = np.exp(-np.cumsum(mu))
  survie_debut = np.concatenate(([1.0], survie_fin[:-1]))
  return np.sum(survie_debut * (-np.expm1(-mu)) / mu)

esperances_brutes = pd.DataFrame(
    {nom: [esperance_brute(tab, x) for x in ages] for nom, tab in jeux.items()},
    index=[f"e{x}" for x in ages]).T

print("\nEspérances de vie brutes observées")
print(esperances_brutes.round(2))
esperances_brutes.round(2).to_csv(RESULTATS / "esperances_vie_brutes.csv")

# Calcul des biais de modélisation (Estimé - Observé)
biais_esperance_mse = (esperances_mse - esperances_brutes).round(2)
biais_esperance_log_like = (esperances_log_like - esperances_brutes).round(2)

print("\nBiais de l'espérance de vie (MSE)")
print(biais_esperance_mse)
biais_esperance_mse.to_csv(RESULTATS / "biais_esperances_mse.csv")

print("\nBiais de l'espérance de vie (Log-Likelihood)")
print(biais_esperance_log_like)
biais_esperance_log_like.to_csv(RESULTATS / "biais_esperances_log_like.csv")

# Graphique comparatif superposé (MSE vs Log-Likelihood) 
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (nom, tab) in zip(axes.flat, jeux.items()):
    ax.plot(tab.age, tab.mu, "ko", ms=3, label="Taux bruts")
    ax.plot(tab.age, mu_modele(tableau_mse.loc[nom], tab.age), "r-", lw=2, label="Modèle MSE")
    ax.plot(tab.age, mu_modele(tableau_log_like.loc[nom], tab.age), "b--", lw=2,
            label="Modèle Log-Likelihood")
    ax.set(xlabel="Âge", ylabel=r"$\mu_x$", title=nom)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
plt.tight_layout()
sauvegarder_figure("graphique_comparatif")

# Graphique comparatif en échelle logarithmique 
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (nom, tab) in zip(axes.flat, jeux.items()):
    ax.plot(tab.age, tab.mu, "ko", ms=3, label="Taux bruts")
    ax.plot(tab.age, mu_modele(tableau_mse.loc[nom], tab.age), "r-", lw=2, label="Modèle MSE")
    ax.plot(tab.age, mu_modele(tableau_log_like.loc[nom], tab.age), "b--", lw=2,
            label="Modèle Log-Likelihood")
    ax.set_yscale("log")
    ax.axvline(70, color="gray", ls=":", lw=1)  # âge du changement de pente (x = 70)
    ax.set(xlabel="Âge", ylabel=r"$\mu_x$ (échelle log)", title=nom)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
plt.tight_layout()
sauvegarder_figure("graphique_comparatif_echelle_log")


# Q3 - Tarification de rentes viagères temporaires et comparaison légale 

#Tarification des rentes viagères à terme échu (fin de période)
# Hypothèses : i = 1.25%, rente de 1000€, durée de 20 ans, âges 50, 60, 70 (femmes 2020 & 2023)
def prix_rente(parametres, x, n=20, i=0.0125, montant=1000):
    v = 1 / (1 + i)
    mu = mu_modele(parametres, np.arange(x, x + n))
    tpx = np.cumprod(np.exp(-mu))
    return montant * np.sum(v ** np.arange(1, n + 1) * tpx)

ages_rentes = [50, 60, 70]
colonnes = [f"x = {x}" for x in ages_rentes]

prix_mse = pd.DataFrame({nom: [prix_rente(tableau_mse.loc[nom], x) for x in ages_rentes]
                         for nom in ["F2020", "F2023"]}, index=colonnes).T
prix_log_like = pd.DataFrame({nom: [prix_rente(tableau_log_like.loc[nom], x) for x in ages_rentes]
                              for nom in ["F2020", "F2023"]}, index=colonnes).T

print("\nPrix des rentes (MSE)")
print(prix_mse.round(2))
prix_mse.round(2).to_csv(RESULTATS / "prix_rentes_mse.csv")

print("\nPrix des rentes (Log-Likelihood)")
print(prix_log_like.round(2))
prix_log_like.round(2).to_csv(RESULTATS / "prix_rentes_log_like.csv")


# Comparaison avec les tables de mortalité légales 
# Référence : table XR (unisexe), chargement de la table XR depuis le fichier CSV du dossier data
xr = pd.read_csv(DOSSIER / "data" / "table_XR.csv")
XR_lx = xr.set_index("age")["lx"]

def lx_XR(x):
    return XR_lx.loc[np.atleast_1d(x)].values.astype(float)

# Prix d'une rente temporaire selon la table légale 
def prix_rente_legale(lx, x, n=20, i=0.0125, montant=1000):
    v = 1 / (1 + i)
    t = np.arange(1, n + 1)
    tpx = lx(x + t) / lx(x)[0]
    return montant * np.sum(v ** t * tpx)


prix_legal_XR = pd.Series([prix_rente_legale(lx_XR, x) for x in ages_rentes], index=colonnes)

# Écarts par rapport à XR 
ecart_mse_XR_percent = 100 * (prix_mse - prix_legal_XR) / prix_legal_XR
ecart_log_like_XR_percent = 100 * (prix_log_like - prix_legal_XR) / prix_legal_XR


# Tableau de synthèse des tarifs
tableau_prix = pd.concat([
    prix_mse.rename(index=lambda n: "MSE_" + n),
    prix_log_like.rename(index=lambda n: "LL_" + n),
    prix_legal_XR.to_frame("Legal_XR").T,
])

print("\nSynthèse comparative des prix des rentes (€)")
print(tableau_prix.round(0))
tableau_prix.to_csv(RESULTATS / "synthese_prix_rentes.csv")

print("\nÉcart relatif au tarif réglementaire XR (%) - Modèle MSE")
print(ecart_mse_XR_percent.round(1))
ecart_mse_XR_percent.to_csv(RESULTATS / "ecarts_rentes_mse_xr_percent.csv")

print("\nÉcart relatif au tarif réglementaire XR (%) - Modèle Log-Likelihood")
print(ecart_log_like_XR_percent.round(1))
ecart_log_like_XR_percent.to_csv(RESULTATS / "ecarts_rentes_loglike_xr_percent.csv")

