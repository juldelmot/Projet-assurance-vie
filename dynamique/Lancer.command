#!/bin/zsh
cd -- "$(dirname -- "$0")"
./.venv/bin/python partie_dynamique.py
statut=$?
echo ""
if [ "$statut" -eq 0 ]; then
    echo "Calcul terminé. Les résultats sont dans le dossier resultats."
else
    echo "Le calcul a rencontré une erreur. Le message est affiché ci-dessus."
fi
read "?Appuie sur Entrée pour fermer."
exit "$statut"
