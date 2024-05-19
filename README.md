# Hodnocení kvality chirurgického stehu pomocí počtu stehů

Cílem této úlohy je vyhodnotit kvalitu chirurgického stehu na základě obrazu incize a stehu.

- ## Příklad spuštění:

```bash
cd GithubProjects/ZDO_Team3/src
python run.py output.csv incision001.jpg incision005.png incision010.JPEG
```

- ## Příklad spuštění s vizualizací:

```bash
python run.py output.csv -v incision001.jpg incision005.png
```

## Struktura výstupního souboru CSV je demonstrována v následujícím příkladu. Hlavička je "filename" a "n_stiches".
```
filename, n_stiches
incision000.jpg , 5 # obrázek obsahuje 5 stehů
incision001.jpg , 2
incision003.jpg , 0
incision002.jpg , -1 # obrázek nemohl být zpracován
```