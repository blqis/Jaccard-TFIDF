# Utilisation

```
python3 -m venv env_idl_16             
source env_idl_16/bin/activate
pip install -r requirements.txt
```

Les tests sont commentés au sein même du code.

`TP8.py`est le fichier principal, et `BONUSnltk.py` en est une copie quasi conforme hormis le module de traitement de corpus qui est différent.


# Préparation du dataset

Il est impératif de tokenizer, normaliser, lemmatiser, et supprimer les mots vides de nos sous-corpus.

J'étais d'abord partie sur nltk poue ce traitement là, mais au vu des performances assez décévantes de lemmatisation j'ai opté pour spaCy. Associer les deux greffe le temps d'exécution, ce pourquoi il a été plus intéréssant de reléguer la pipeline entière à un seul module.

# Calcul de la représentation vectorielle

####  Quelle doit-être la taille des vecteurs représentant chaque document de notre corpus ?

La taille des vecteurs doit être égale à la taille du vocabulaire de notre corpus. Cette valeur peut être obtenue en additionnant toutes les longueurs des valeurs du dictionnaire après le pré-traitement.

# Résultats

## Recommendation

```
Les recommandations basées sur la mesure de similarité de Jaccard pour le film "The Nightmare Before Christmas" sont :
 [('The Best Years of Our Lives', 0.13636363636363635), ('Finding Nemo', 0.08695652173913043), ('A Christmas Story', 0.08)]
```
```
Les recommandations basées sur la mesure de similarité de Jaccard pour le film "The Nightmare Before Christmas" sont :
 [('The Best Years of Our Lives', 0.17578118939111298), ('A Christmas Story', 0.13492021519567485), ('Life of Brian', 0.1194314686105238)]
```

Pour le film *The Godfather*, le corpus traité avec spaCy génère bien le sequel *The Godfather: Part II* comme TOP3 proposition de film avec la fonction `recommend_TFIDF`, là où celui traité avec nltk le place à la 5e position. Des résultats un peu plus concluants avec spaCy, donc !