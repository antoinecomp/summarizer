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

Here We try the attempt provided by [prijatelj](https://stackoverflow.com/a/38554331/4764604). That is to say :

 1. I prep the image: as far as my images I want to extract the text from have roughly rougly two types of writings, by hand of a range of gray colors, and the text which is always black. I would first white out all content that is not black (or already white). Doing so will leave only the black text left. 
 2. Now that all you have is the black text the goal is to create boxes. As stated by prijatelj, there are different ways of going about this. Here I try his Home Brewed Non-SWT Method:
   a. I erode image based on given kernel size (erosion = expands black areas)
   b. finds contours of eroded image (but do I have contours in my case ?)
   c. finds bounding boxes of all contours

It seems I should be able to get the black text boxes coordinates thanks to this approach. I want to do the same with the grey approach and get the black text from the coordinates of the grey text.

# To-do

implement the grouping algorithm that would group and seperate strokes and text


# Inspiration

As I want to distinguish shape versus text in hand-drawn strokes [Using Entropy to Distinguish Shape Versus Text in Hand-Drawn Diagrams](https://www.ijcai.org/Proceedings/09/Papers/234.pdf) rose my interest. Yet it doesn't tell what the grouping algorithm they use to group only strokes which are a part of the drawings or the letter.
