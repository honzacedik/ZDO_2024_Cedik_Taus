# Hodnocení kvality chirurgického stehu pomocí počtu stehů

Cílem této úlohy je vyhodnotit kvalitu chirurgického stehu na základě obrazu incize a stehu.

- ## Příklad spuštění:

```bash
cd GithubProjects/ZDO_Team3/src
python run.py output.csv incision001.jpg incision005.png incision010.JPEG
```

- # Pokud nebudou na vstupu žádné argumenty, pustí se ukázková verze na předdefinavém obrázku

- ## Příklad spuštění s vizualizací:

```bash
python run.py output.csv -v incision001.jpg incision005.png
```

## Struktura výstupního souboru CSV je demonstrována v následujícím příkladu. Hlavička je "filename" a "Počet_stehů".
```
filename, Počet_stehů
incision000.jpg , 5 # obrázek obsahuje 5 stehů
incision001.jpg , 2
incision003.jpg , 0
```
