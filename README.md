# summarizer

This summarizer intends to sum up pictures of annotated pdfs. 
It should group together all strokes which are in a spatial threshold and select only the siding text.
For instance with the following image (red arrows are not part of the image and are here to show the strokes) :

![image of a text with strokes on its left](https://i.stack.imgur.com/9xedG.png)

We should sum it up as :

```
A l'opposé le Nord et l'Est de la Seine-Saint-Denis cumulent les nombreux handicaps sociaux et résidentiels. On retrouve aussi ces difficultés en deuxième couche de la basse Seine (Les Mureaux, Mantes-La-Jolie), dans certaines villes nouvelles (Cergy, Trappes, Evry, Grigny) et dans les villes secondaires,

Les processus de renforcement des ségrégations concernent aujourd'hui l'ensemble de l'Île-de-France comme l'indique la comparaison départementale. Du fait des blocages sociaux et résidentiels de ces dernières

Du fait de l'énorme bulle immobilière spéculative des dernières décennies, qui touche en particulier Paris et une partie de la première couronne, mais qui se répercute mécaniquement sur l'ensemble de l'espace régional
```

# To-do

implement the grouping algorithm that would group and seperate strokes and text


# Inspiration

As I want to distinguish shape versus text in hand-drawn strokes [Using Entropy to Distinguish Shape Versus Text in Hand-Drawn Diagrams](https://www.ijcai.org/Proceedings/09/Papers/234.pdf) rose my interest. Yet it doesn't tell what the grouping algorithm they use to group only strokes which are a part of the drawings or the letter.
