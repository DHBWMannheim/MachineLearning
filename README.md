@patrick

# Machine Learning

Dieses Notebook beinhaltet die Prüfungsleistung im Kurs WWI18SEA/C für Machine Learning bei Frau Minges. Es wurde erstellt von Patrick Mischka, Jan Grübener, Matthias Vonend, Aaron Schweig, Michael Angermeier und Troy Keßler.

Hinweis: Alle Sektionen, welche mit @ annotiert sind, sind lediglich Einteilungen für die Präsentation und stellen nicht die Leistung der Personen dar.

## Inhalt

Ziel ist mithilfe von Machine Learning eine Trading Recommendation für Ethereum zu entwickeln. Diese soll aus zwei Teilen bestehen, zum einen aus einer technischen Analyse, wo ein LSTM Modell unter Verwendung von historischen Preisen und Indikatoren entwickelt und trainiert wird, und zum Anderen aus einer Stimmungsanalyse auf Twitter, wofür ein weiteres Deep Neural Network entwickelt und trainiert wird. Letztendlich sollen die Ergebnisse dieser Modelle Entscheidungshilfen sein, um Ethereum zu handeln.

### 1. Sentimentmodell

Im ersten Teil wird ein Sentimentmodell entwickelt und trainiert. Das Ziel ist hier ein Modell zu entwickelt, welches Tweets auf ihre Stimmung bewerten kann.

### 2. Technisches Modell

Im zweiten Teil wird ein technisches Modell entwickelt und trainiert. Das Ziel ist hier, basierend auf historischen Preisverläufen und anderen technischen Indikatoren, den zukünftigen Preis für die nächsten 30 Tage vorherzusagen.

### 3. Ausführung

Im dritten und letzten Teil werden die Modelle an APIs angeschlossen, so dass die Entscheidungshilfen live ausgeführt werden können.

## Technologien

Für das Modell wird [Tensorflow](https://www.tensorflow.org/) verwendet, zum Plotten von Informationen nutzen wir [Matplotlib](https://matplotlib.org/stable/index.html) und zum Verarbeiten von Daten [Pandas](https://pandas.pydata.org/). Außerdem werden weitere Utilities von [sklearn](https://scikit-learn.org/stable/) übernommen.

## Setup

Um dieses Notebook zu benutzen müssen Python 3.x (vorzugsweise 3.7.3) und folgende Packages installiert werden:

* tensorflow==2.4.1
* matplotlib==3.3.4
* pandas==1.2.2
* pandas_datareader==0.9.0
* searchtweets-v2==1.0.7
* scikit-learn==0.24.1
* seaborn==0.11.0
* numpy==1.19.2

Diese können auch automatisch über die requirements.txt mit pip installiert werden.

Das Datenset für das Trainieren kann über [diesen Link](https://www.dropbox.com/s/ur7pw797mgcc1wr/tweets.csv?dl=0) heruntergeladen werden. Dabei muss die Datei "tweets.csv" in den gleichen Ordner wie dieses Notepad abgelegt werden.


## 1. Sentimentmodell

In diesem Notebook wird ein Modell trainiert, welches Tweets live auf ihre Stimmung bewerten soll. Dafür wird ein Deep Neural Network erstellt, welches mit 1.6 Millionen Tweets trainiert wird. Hierbei handelt es sich um ein Klassifikationsproblem, es soll letztendlich entschieden werden, ob ein Tweet negativ (0), oder positiv (1) gestimmt ist.

### Datensatz

Um nun das Modell möglichst gut darauf zu trainieren reale Tweets zu bewerten haben wir uns für ein Datenset entschieden, welches 1.6 Millionen bereits gelabelte Tweets enthält. Dieses Datenset kann [hier](https://www.kaggle.com/kazanova/sentiment140) gefunden werden.


```python
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow import feature_column
```

### Laden des Datensatzes

Mithilfe von pandas wird das Datenset geladen. Dabei werden nur die erste und die letzte Spalte geladen, da nur diese für uns von Interesse sind. Da es sich bei der ersten Spalte um die Stimmung des Tweets handelt, wird diese mit "targets" gelabelt. Die letzte Spalte beinhaltet den eigentlichen Tweet, welcher mit "text" gelabelt wird.


```python
dataframe = pd.read_csv("./tweets.csv", usecols=[0, 5], names=["target", "text"])
```

Da das Datenset sortiert ist, muss es randomisiert werden. Falls dies nicht gemacht werden würde, hätte dies einen negativen Einfluss auf das Lernen. Zuerst würden alle negativ gelabelten Daten geladen werden, wodurch das Modell "denkt", dass alle Daten negativ wären. Das Modell würde sich entsprechend darauf einstellen. Werden positiven Daten verwerdet, würde das Modell annehmen, dass es nur positive Daten gäbe. Dementsprechend würde es bei richtigen (nicht-trainings) Daten immer eine positive Stimmung vorhersagen, was aber nicht der Realtität entsprechen würde.


```python
dataframe = shuffle(dataframe)
```

Wenn der Datensatz korrekt geladen wurde sollte eine Tabelle mit den ersten fünf Einträgen zu sehen sein.


```python
dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>873380</th>
      <td>4</td>
      <td>Ahhhhhh....2 jumps- I feel back to center agai...</td>
    </tr>
    <tr>
      <th>643814</th>
      <td>0</td>
      <td>I really do want to go to sleep, but I can't s...</td>
    </tr>
    <tr>
      <th>618262</th>
      <td>0</td>
      <td>@callmedingding don't laugh at me jerk. i hurt</td>
    </tr>
    <tr>
      <th>293885</th>
      <td>0</td>
      <td>@LaurenConrad im crying the hills is nuthing w...</td>
    </tr>
    <tr>
      <th>478219</th>
      <td>0</td>
      <td>fuck, i've lost my wallet.. I think</td>
    </tr>
  </tbody>
</table>
</div>



Um das Trainieren des Modells zu überwachen und um die Trefferquote des Modells hinterher zu errechnen wird der Datensatz in drei Teile unterteilt. In einem Verhältnis von 80:20 wird der Datensatz in Trainingsdaten und Testdaten unterteilt. Trainingsdaten dienen hier ausschließlich zum Trainieren des Modells. Die Testdaten werden nach dem Trainieren dazu verwendet, um die Trefferquote des Modells zu errechnen. Diese sollen reale Daten simulieren. Dieses Verhältnis wurde gewählt, da mehr Trainingsdaten ein besseres Ergebnis versprechen. Die Anzahl der Testdaten muss hingegen nicht hoch sein, um die Trefferquote zu bestimmen.

Weiterhin werden die Trainingsdaten wiederum in Trainingsdaten und Validierungsdaten aufgeteilt. Auch hier wird ein Verhältnis von 80:20 angesetzt. Die Validierungsdaten werden dazu verwendet, um das Training zu überwachen. Nach jeder Epoche (Trainingsschritt) wird damit die aktuelle Trefferquote bestimmt.


```python
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'training tweets')
print(len(val), 'validation tweets')
print(len(test), 'test tweets')
```

    1024000 training tweets
    256000 validation tweets
    320000 test tweets
    

Da jetzt der Datensatz entsprechend aufgeteilt wurde, kann er nun in das Tensorflow-Format gebracht werden. Dafür werden die Features (text) und die Labels (labels) klar definiert. Zusätzlich wird eine Batchgröße definiert, welche Daten gruppiert und dadurch das Lernen beschleunigt.


```python
def df_to_dataset(dataframe, batch_size):
  dataframe = dataframe.copy()
  texts = dataframe.pop('text')
  labels = dataframe.pop('target')
  return tf.data.Dataset.from_tensor_slices((texts, labels)).batch(batch_size)
```


```python
batch_size = 320

raw_train_ds = df_to_dataset(train, batch_size)
raw_val_ds = df_to_dataset(val, batch_size)
raw_test_ds = df_to_dataset(test, batch_size)
```

Um zu validieren, dass die Konvertierung erfolgreich war, werden die ersten drei Einträge ausgelesen.


```python
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Tweet:", text_batch.numpy()[i])
    print("Label:", label_batch.numpy()[i])
```

    Tweet: b"@grossdale have fun in fl. I'll be there the day you leave "
    Label: 0
    Tweet: b'needs a full day of sleep ... like 20 hrs instead of 3-4 hrs a night... starting to catch up!!!  '
    Label: 0
    Tweet: b"why are you such an idiot ? i can't understand u, mygooosh! "
    Label: 0
    

@matthias

Um unnötige Duplikate zu vermeiden, werden die Daten für das Modell normalisiert. Beispielsweiße werden Wörter, die in manchen Tweets groß und in anderen wieder klein geschrieben werden, angepasst. Zusätzlich können User-Namen, welche mit "@" beginnen normalisiert werden, da der genaue User-Name unwichtig für die Sentiment-prediction ist.


```python
def normalize_data(tweet):
  lowercase = tf.strings.lower(tweet)
  return tf.strings.regex_replace(lowercase, '@(\w*)|(\\n)|(https:\/\/t\.co[\w\/]*)', '')
```

Nun können die Texte vektorisiert werden. Da ein neuronales Netz nicht mit Wörtern und Buchstaben arbeiten kann, müssen diese in Zahlen umgewandelt werden. Dafür werden die Tweets in Vektoren umgewandelt. Die Größe des Vektors wird dabei mit sequence_length definiert. Die Größe der sequence_length, also die Größe des Vektors, sollte in der Regel so groß sein, dass alle Wörter eines Tweets hineinpassen. Da die Anzahl an Zeichen auf 280 pro Tweet limitiert ist, und die durchnittliche Anzahl der Zeichen pro Wort im Englischen bei 5 liegt, wird die sequence_length mit 56 definiert.

Hier erhält jedes Wort eine fortlaufende Id. Die Reihenfolge dieser Ids ist durch die Reihenfolge in dem die Wörter vektorisiert wurden festgelegt. Dabei können aufgrund dictionary_size maximal 10000 Wörter eingelesen werden. Alle weiteren Wörter werden ignoriert. Diese Menge an Vokabeln sollte aber ausreichen, da in der Alltagssprache lediglich zwei- bis dreitausend Wörter verwendet werden. Somit kann jedes Wort einer Id zugewiesen werden, sodass man ganze Sätze in einem Vektor abbilden kann. Da die Vektorengröße immer der sequence_length enstpricht, wird auch das Problem, dass ein neuronales Netz immer die gleiche Inputgröße benötigt, gelöst.

Dafür wird hier ein Vektorlayer erstellt. Gleichzeitig können hier die Daten normalisiert werden.


```python
dictionary_size = 10000
sequence_length = 56

vectorize_layer = TextVectorization(
    standardize=normalize_data,
    max_tokens=dictionary_size,
    output_mode='int',
    output_sequence_length=sequence_length)
```

Hier werden die Trainingsdaten eingelesen, so dass die 10000 Features gefüllt werden können. Es entsteht ein "Wörterbuch" für Tweets


```python
# train_text = raw_train_ds.map(lambda x, y: x)
train_text = np.concatenate([x for x, y in raw_train_ds], axis=0)
vectorize_layer.adapt(train_text)
```

Mit der Methode können wir alle Datensätze vektorisieren. Hier normalisieren wir das Label, so dass das Label eine Wertebereich von 0 bis 1, anstatt von 0 bis 4 besitzt. 


```python
def vectorize_tweet(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), int(label / 4)
```

Um zu testen, ob das Vektorisieren der Tweets funktioniert, können wir den ersten Tweet aus dem ersten Batch auslesen und vektorisieren.


```python
text_batch, label_batch = next(iter(raw_train_ds))
text, label = text_batch[0], label_batch[0]
print(text)
print(label)
print(vectorize_tweet(text, label))
```

    tf.Tensor(b"@grossdale have fun in fl. I'll be there the day you leave ", shape=(), dtype=string)
    tf.Tensor(0, shape=(), dtype=int64)
    (<tf.Tensor: shape=(1, 56), dtype=int64, numpy=
    array([[ 16, 136,  11,   1, 108,  24,  96,   4,  42,   8, 349,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0]], dtype=int64)>, 0)
    

Mithilfe des Vektorlayers können wir von den Ids wieder auf die Wörtern zurückschließen. Außerdem können wir die Größe unseres Wörterbuchs auslesen.


```python
print("Token 1234:", vectorize_layer.get_vocabulary()[1234])
print('Dictionary size: {}'.format(len(vectorize_layer.get_vocabulary())))
```

    Token 1234: comment
    Dictionary size: 10000
    

Nun vektorisieren wir alle benötigten Datensätze.


```python
train_ds = raw_train_ds.map(vectorize_tweet)
val_ds = raw_val_ds.map(vectorize_tweet)
test_ds = raw_test_ds.map(vectorize_tweet)
```

Aus Performancegründen können die Datensätze weiter aufbereitet werden. Mit `.cache()` bleiben die Daten im Arbeitsspeicher, nachdem diese von der Festplatte geladen wurden. Somit kann sichergestellt werden, dass das Laden der Daten nicht der Flaschenhals beim Training sein wird.

Mit `.prefetch()` können die Daten gleichzeitig mit dem Lernen präprozessiert werden.


```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

Schließlich definieren wir das eigentliche Modell. Der erste Layer ist ein Embedding-Layer. Dieser sorgt dafür, dass jedes Wort einen eigenen Vektor erhält, welcher die Bedeutung des Wortes darstellt. Diese Vektoren werden mit dem Modell über die Zeit trainiert. Dieser Embedding-Layer fügt eine weitere Dimension zum Outputvektor hinzu. Hier definieren wir mit der Embedding-Dimension die Größe der Layer, das bedeutet, dass es 32 Nodes pro Layer gibt.

Als nächster Layer wird `GlobalAveragePooling1D` verwendet. Dieser reduziert die Dimension wieder um 1 und verrechnet dabei alle Informationen, sodass keine Informationen verloren gehen. Der Outputvektor wird dabei wieder auf eine feste Länge normalisiert.

Anschließend folgt ein fully-connected 32 Dense-Layer. Hier wurde eine Dropoutrate festgelegt, um Overfitting zu verhindern. Das Ziel hier ist random ausgewählte Nodes auf 0 zu setzen, damit das anspassen der Weights der einzelnen Nodes beim Lernen gefördert wird.

Letztendlich wird der letzte Layer mit einem Dense Layer zu einer einzigen Node verknüpft. Diese hat dank der Sigmoid-Aktivierungsfunktion ein Intervall von 0 bis 1 und gibt das Ergenis aus.

Wir können nun noch mit `.summary()` das Modell verifizieren.


```python
model = tf.keras.Sequential([
  layers.Embedding(dictionary_size, 32),
  layers.GlobalMaxPooling1D(),
  layers.Dropout(0.1),
  layers.Dense(1, activation='sigmoid')
])

model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, None, 32)          320000    
    _________________________________________________________________
    global_max_pooling1d (Global (None, 32)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 320,033
    Trainable params: 320,033
    Non-trainable params: 0
    _________________________________________________________________
    

@troy

Für das Trainieren müssen noch ein paar Parameter definiert werden. Für die Berechnung des Fehlers (loss) verwenden wir die `BinaryCrossentropy` Funktion. Der Fehler gibt uns an, wie weit wir von der richtigen Prediction weg sind. Wir haben uns dafür entschieden, da wir einen sogenannten Binary Classifier haben, der uns eine Wahrscheinlichkeit von 0 bis 1 als Ergebnis gibt.

Weiterhin verwenden wir für den Optimierungsalgorithmus den `Adam-Optimizer`. Wir haben uns für den Adam-Optimizer, im Vergleich zum klassischen Stochastic-Gradient-Descent-Algorithmus entschieden, da sich die Learningrate beim Adam-Optimizer mit der Zeit automatisch anpasst. Das ist besonders praktisch bei Natural-Language-Processing, da hier die Gradients in der Regel sehr gering sind. Dabei wird die Learningrate basierend auf der vorherigen Änderung der Weights angepasst. Hier haben wir eine sehr kleine Learningrate definiert, da wir ein sehr großes Datenset haben und nicht zu schnell in das Problem von Overfitting laufen wollen.

Zusätzlich werden weitere Metriken wie True Positives, False Positives, True Negatives, False Negatives, Precision, Recall und AUC gemessen, um genauere Aussagen über die Genauigkeit des Modells zu treffen.


```python
metrics = [
  tf.metrics.TruePositives(name='tp'),
  tf.metrics.FalsePositives(name='fp'),
  tf.metrics.TrueNegatives(name='tn'),
  tf.metrics.FalseNegatives(name='fn'), 
  tf.metrics.BinaryAccuracy(name='accuracy'),
  tf.metrics.Precision(name='precision'),
  tf.metrics.Recall(name='recall'),
  tf.metrics.AUC(name='auc'),
]

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=metrics)
```

Nun wird das Modell trainiert. Dafür definieren wir mit epochs, wie oft wir über das Trainingsdatenset iterieren. In `model.fit()` werden die Trainingsdaten, die Validationsdaten und die Anzahl der Epochen angegeben. Tensorflow loggt den Fortschritt live in der Konsole aus und zusätzlich wird der Trainingsstatus in einem History-Objekt festgehalten.


```python
training_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10)
```

    Epoch 1/10
    3200/3200 [==============================] - 31s 9ms/step - loss: 0.6577 - tp: 156518.8166 - fp: 56567.5114 - tn: 199717.6001 - fn: 99515.9719 - accuracy: 0.6709 - precision: 0.7073 - recall: 0.5846 - auc: 0.7300 - val_loss: 0.5372 - val_tp: 94921.0000 - val_fp: 29195.0000 - val_tn: 98917.0000 - val_fn: 32967.0000 - val_accuracy: 0.7572 - val_precision: 0.7648 - val_recall: 0.7422 - val_auc: 0.8308
    Epoch 2/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.5185 - tp: 194603.1793 - fp: 58898.5345 - tn: 197386.5770 - fn: 61431.6092 - accuracy: 0.7634 - precision: 0.7668 - recall: 0.7564 - auc: 0.8369 - val_loss: 0.4819 - val_tp: 99400.0000 - val_fp: 29005.0000 - val_tn: 99107.0000 - val_fn: 28488.0000 - val_accuracy: 0.7754 - val_precision: 0.7741 - val_recall: 0.7772 - val_auc: 0.8537
    Epoch 3/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4748 - tp: 200245.1097 - fp: 57128.4427 - tn: 199156.6689 - fn: 55789.6789 - accuracy: 0.7789 - precision: 0.7771 - recall: 0.7815 - auc: 0.8575 - val_loss: 0.4623 - val_tp: 100345.0000 - val_fp: 28019.0000 - val_tn: 100093.0000 - val_fn: 27543.0000 - val_accuracy: 0.7830 - val_precision: 0.7817 - val_recall: 0.7846 - val_auc: 0.8638
    Epoch 4/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4565 - tp: 201891.4549 - fp: 54924.4136 - tn: 201360.6979 - fn: 54143.3336 - accuracy: 0.7869 - precision: 0.7856 - recall: 0.7886 - auc: 0.8675 - val_loss: 0.4528 - val_tp: 100645.0000 - val_fp: 27281.0000 - val_tn: 100831.0000 - val_fn: 27243.0000 - val_accuracy: 0.7870 - val_precision: 0.7867 - val_recall: 0.7870 - val_auc: 0.8691
    Epoch 5/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4464 - tp: 202485.0519 - fp: 53292.8750 - tn: 202992.2365 - fn: 53549.7366 - accuracy: 0.7911 - precision: 0.7911 - recall: 0.7906 - auc: 0.8732 - val_loss: 0.4473 - val_tp: 100721.0000 - val_fp: 26530.0000 - val_tn: 101582.0000 - val_fn: 27167.0000 - val_accuracy: 0.7902 - val_precision: 0.7915 - val_recall: 0.7876 - val_auc: 0.8722
    Epoch 6/10
    3200/3200 [==============================] - 29s 9ms/step - loss: 0.4398 - tp: 202980.9925 - fp: 52181.3374 - tn: 204103.7741 - fn: 53053.7960 - accuracy: 0.7945 - precision: 0.7951 - recall: 0.7930 - auc: 0.8770 - val_loss: 0.4439 - val_tp: 100828.0000 - val_fp: 26177.0000 - val_tn: 101935.0000 - val_fn: 27060.0000 - val_accuracy: 0.7920 - val_precision: 0.7939 - val_recall: 0.7884 - val_auc: 0.8742
    Epoch 7/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4348 - tp: 203462.0075 - fp: 51213.2596 - tn: 205071.8519 - fn: 52572.7810 - accuracy: 0.7972 - precision: 0.7984 - recall: 0.7947 - auc: 0.8799 - val_loss: 0.4417 - val_tp: 100953.0000 - val_fp: 25932.0000 - val_tn: 102180.0000 - val_fn: 26935.0000 - val_accuracy: 0.7935 - val_precision: 0.7956 - val_recall: 0.7894 - val_auc: 0.8755
    Epoch 8/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4310 - tp: 203642.1824 - fp: 50673.6226 - tn: 205611.4889 - fn: 52392.6061 - accuracy: 0.7987 - precision: 0.8003 - recall: 0.7956 - auc: 0.8820 - val_loss: 0.4402 - val_tp: 101019.0000 - val_fp: 25790.0000 - val_tn: 102322.0000 - val_fn: 26869.0000 - val_accuracy: 0.7943 - val_precision: 0.7966 - val_recall: 0.7899 - val_auc: 0.8764
    Epoch 9/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4285 - tp: 203954.6364 - fp: 50278.4149 - tn: 206006.6967 - fn: 52080.1521 - accuracy: 0.8001 - precision: 0.8019 - recall: 0.7967 - auc: 0.8834 - val_loss: 0.4391 - val_tp: 101041.0000 - val_fp: 25658.0000 - val_tn: 102454.0000 - val_fn: 26847.0000 - val_accuracy: 0.7949 - val_precision: 0.7975 - val_recall: 0.7901 - val_auc: 0.8770
    Epoch 10/10
    3200/3200 [==============================] - 30s 9ms/step - loss: 0.4259 - tp: 204165.2765 - fp: 49650.7610 - tn: 206634.3505 - fn: 51869.5120 - accuracy: 0.8018 - precision: 0.8041 - recall: 0.7975 - auc: 0.8849 - val_loss: 0.4384 - val_tp: 101060.0000 - val_fp: 25578.0000 - val_tn: 102534.0000 - val_fn: 26828.0000 - val_accuracy: 0.7953 - val_precision: 0.7980 - val_recall: 0.7902 - val_auc: 0.8775
    

Mithilfe von Matplotlib können wir den Loss plotten und beobachten, wie sich diese während des Lernens verhalten hat. Optimalerweise sollte diese mit der Zeit sinken, da mit dem Anpassen der Weights das Modell immer genauere Aussagen treffen sollte und somit auch der Fehler immer geringer werden sollte.

Wir können erkennen, dass dies tatsächlich der Fall ist. Der Loss fällt fast exponentiell. Logischerweise wird der Trainingsloss immer geringer. Als Bestätigung für die Verbesserung des Modells dient hier der Validationloss. Dieser ist fast gleich, sodass wir davon ausgehen können, dass die Anzahl der Fehlinterpretierungen tatsächlich geringer wurde.


```python
history_dict = training_history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'o', color='orange', label='Training Loss')
plt.plot(epochs, val_loss, 'blue', label='Validation Loss')
plt.title('Trainings- und Validationsloss')
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


    
![png](output_39_0.png)
    


Das Gleiche können wir auch für die Accuracy durchführen. Hier sollte im Optimalfall die Accuracy mit der Zeit steigen. Dieses Verhalten können wir wieder an unserem Modell erkennen. Hier erinnert der Graph an eine Sättigungskurve. Dies liegt daran, dass das Lernen letztendlich eine Optimierung ist und es mit der Zeit immer schwerer wird, das Modell noch mehr zu verbessern.

An beiden Graphiken kann man jedoch gut erkennen, dass es zu keinem Overfitting kommt. Wenn wir die Accuracy betrachten, würde bei Overfitting die Accuracy der Testdaten weiter ansteigen, während die Accuracy der Validationsdaten und die der Testdaten stagniert oder gar sinken. Das Gleiche würde analog mit dem Loss passieren.


```python
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'o', color='orange', label='Training Accuracy')
plt.plot(epochs, val_acc, 'blue', label='Validation Accuracy')
plt.title('Trainings- und Validationsaccuracy')
plt.xlabel('Epochen')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```


    
![png](output_41_0.png)
    


Nachdem das Modell nun erfolgreich trainiert wurde können wir es mit den vorher festgelegten Testdatensatz testen. Zuerst extrahieren wir noch die Features und Labels aus den Datasets, um mit ihnen arbeiten zu können.


```python
train_features = np.concatenate([x for x, y in train_ds], axis=0)
train_labels = np.concatenate([y for x, y in train_ds], axis=0)

test_features = np.concatenate([x for x, y in test_ds], axis=0)
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

train_predictions_baseline = model.predict(train_features, batch_size=batch_size)
test_predictions_baseline = model.predict(test_features, batch_size=batch_size)
```

Mit den vorher gemessenen Metriken kann nun die Genauigkeit mithilfe einer Confusion-Matrix bestimmt werden. Hier wird ein Threshold von 0.5 verwendet, um die Anzahl an falschen Predictions insgesamt zu reduzieren. Dabei hat das Modell eine Testaccuracy von 78.23%, während die Präzision bei 76.93% und der Recall bei 80.43% liegt. Die Präzision sagt hier aus, dass 76.93% aller positiv predicteten Tweets tatsächlich positiv waren. Der Recall sagt hier, dass 80.43% aller positiven Tweets korrekt klassifiziert wurden.

Weiterhin beträgt die AUC 85.52% und da die Kurve über der Diagonale liegt ist das Modell besser als der Zufall.


```python
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Negative Stimmung korrekt erkannt (True Negatives): ', cm[0][0])
  print('Negative Stimmung falsch positiv erkannt (False Positives): ', cm[0][1])
  print('Positive Stimmung falsch negativ erkannt (False Negatives): ', cm[1][0])
  print('Positive Stimmung korrekt erkannt (True Positives): ', cm[1][1])

baseline_results = model.evaluate(test_features, test_labels, batch_size=batch_size, verbose=0)

for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)

print()

plot_cm(test_labels, test_predictions_baseline)
```

    loss :  0.4370047450065613
    tp :  126839.0
    fp :  31712.0
    tn :  128061.0
    fn :  33388.0
    accuracy :  0.7965624928474426
    precision :  0.7999886274337769
    recall :  0.7916206121444702
    auc :  0.8783805966377258
    
    Negative Stimmung korrekt erkannt (True Negatives):  128061
    Negative Stimmung falsch positiv erkannt (False Positives):  31712
    Positive Stimmung falsch negativ erkannt (False Negatives):  33388
    Positive Stimmung korrekt erkannt (True Positives):  126839
    


    
![png](output_45_1.png)
    


Die ROC-Kurve beschreibt hier Confusion Matrizen mit unterschiedlichen Thresholds von 0 bis 1. Dabei wird insbesondere das Verhältnis zwischen True Positives und False Positives betrachtet.

(0,0) besagt, alle positiven Tweets wurden korrekt predicted, alle negativen falsch

(1,1) besagt, alle negativen Tweets wurden korrekt predicted, alle positiven falsch

Wie bereits erwähnt beträgt die AUC 85.52% und da die Kurve über der Diagonale liegt ist das Modell besser als der Zufall, sowohl im Training, als auch im Test.


```python
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.grid(True)

plot_roc("Train", train_labels, train_predictions_baseline, color='b')
plot_roc("Test", test_labels, test_predictions_baseline, color='orange', linestyle='--')
plt.plot([0, 1], linewidth=1, linestyle='--')
plt.legend(loc='lower right')
plt.show()
```


    
![png](output_47_0.png)
    


Nun exportieren wir das fertige Modell. Da wir vorher die Texte vektorisiert haben, bevor sie in das Modell gegeben wurden, können wir hier ein Modell exportieren, welche die Texte beim Input vektorisiert. Dies macht uns das zukünftige Predicten einfacher, da das Model nicht immer neu trainiert werden muss. Zusätzlich fügen wir am Ende eine weitere Node mit einer Sigmoid Aktivierungsfunktion hinzu. Diese bildet alle Werte auf Werte zwischen 0 und 1 ab, sodass unsere definiert Sentiment-Range eingehalten wird. Der Vektorisationlayer und die Sigmoid Node wurden beim Lernen vernachlässigt, damit die Lerneffizienz höher ausfällt.


```python
sentiment_model = tf.keras.Sequential([
  vectorize_layer,
  model
])
```

Schließlich können wir einige Beispiele eingeben, um zu sehen, wie sich das Modell verhält. Dabei ist der erste Satz positiv, der zweite neutral und der letzte negativ. Während der neutrale Satz mit rund 0.5 gewertet wird, wird der positive höher gewertet und der negative geringer.


```python
examples = [
  "Today is a great day!",
  "This sentence is rather neutral",
  "This show is terrible!"
]

sentiment_model.predict(examples)
```




    array([[0.9744337 ],
           [0.5660003 ],
           [0.03791037]], dtype=float32)



@jan

## 2. Technisches Modell

Bei dem zweiten Modell soll mithilfe von Finanzdaten eine Progrose erstellt werden, wie der Kursverlauf in den nächsten 30 Tagen sein wird. Für diese Progrose wird ein LSTM-Modell verwendet. Die Prognose wird anhand von des Kurses von Etherium zum US-Dollar aufgebaut. 

### Datensatz

Die Daten zum Trainieren des Modelles werden von Yahoo abgefragt. Bei diesen Daten handelt es sich stets um den "Closing Price", also den Preis, den Etherium am Ende eines Tages hatte. Diese Preise werden bis in das Jahr 2015 geladen. Dies entspricht insgesamt ca. 2000 Preisdaten. Zusätzlich zu den Preisdaten werden mithilfe von der "Technical Analysis Library" verschiedene technische Indikatoren berechnet. Anhand von diesen Indikatoren soll das Modell trainiert werden den Preis vorherzusagen.  


```python
import pandas_datareader.data as pdr

from sklearn.preprocessing import MinMaxScaler
import random

from datetime import datetime, timezone,timedelta
from ta.utils import dropna
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import KAMAIndicator, PercentagePriceOscillator, PercentageVolumeOscillator, ROCIndicator, RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
```

### Laden und generieren der Daten
Im Folgenden werden die Preisdaten von Etherium von 2015 bis heute geladen. Anschließend werden 7 technische Indikatoren generiert und dem Datensatz hinzugefügt. Diese Indikatoren sind:

* Kama
* Percentage Price Oscillator
* Rate of Change
* Moving Average Convergence/Divergence
* Relative Strength Index
* Aaron Indicator
* Bollinger bands

Zusätzlich werden der Tag des Monats, der Tag der Woche und der Monat als eigene Indikatoren hinzugefügt. Dies ist sinnvoll, da hier ein LSTM-, also ein Zeitreihen-Modell verwendet wird. Die Zeit spielt eine wichtige Rolle, um Besonderheiten an beispielsweise dem ersten Tag eines Monats erkennen zu können. Außerdem werden die Daten nicht gemischt, da sonst der zeitliche Verlauf verloren geht.


```python
batch_size = 31
symbol = 'ETH-USD'

end = datetime.today()
start = datetime(2000, 9, 1)
ETH = pdr.DataReader(symbol,'yahoo',start,end)

df = pd.DataFrame(data=ETH)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-06</th>
      <td>3.536610</td>
      <td>2.521120</td>
      <td>2.831620</td>
      <td>2.772120</td>
      <td>1.643290e+05</td>
      <td>2.772120</td>
    </tr>
    <tr>
      <th>2015-08-07</th>
      <td>2.798810</td>
      <td>0.714725</td>
      <td>2.793760</td>
      <td>0.753325</td>
      <td>6.741880e+05</td>
      <td>0.753325</td>
    </tr>
    <tr>
      <th>2015-08-08</th>
      <td>0.879810</td>
      <td>0.629191</td>
      <td>0.706136</td>
      <td>0.701897</td>
      <td>5.321700e+05</td>
      <td>0.701897</td>
    </tr>
    <tr>
      <th>2015-08-09</th>
      <td>0.729854</td>
      <td>0.636546</td>
      <td>0.713989</td>
      <td>0.708448</td>
      <td>4.052830e+05</td>
      <td>0.708448</td>
    </tr>
    <tr>
      <th>2015-08-10</th>
      <td>1.131410</td>
      <td>0.663235</td>
      <td>0.708087</td>
      <td>1.067860</td>
      <td>1.463100e+06</td>
      <td>1.067860</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-04-13</th>
      <td>2449.687500</td>
      <td>2284.563721</td>
      <td>2299.347900</td>
      <td>2435.104980</td>
      <td>3.559282e+10</td>
      <td>2435.104980</td>
    </tr>
    <tr>
      <th>2021-04-14</th>
      <td>2544.267334</td>
      <td>2409.924072</td>
      <td>2436.034668</td>
      <td>2519.116211</td>
      <td>3.232561e+10</td>
      <td>2519.116211</td>
    </tr>
    <tr>
      <th>2021-04-15</th>
      <td>2547.555664</td>
      <td>2318.675049</td>
      <td>2516.601807</td>
      <td>2431.946533</td>
      <td>3.619693e+10</td>
      <td>2431.946533</td>
    </tr>
    <tr>
      <th>2021-04-16</th>
      <td>2497.385254</td>
      <td>2333.682861</td>
      <td>2429.980957</td>
      <td>2344.895020</td>
      <td>3.234981e+10</td>
      <td>2344.895020</td>
    </tr>
    <tr>
      <th>2021-04-18</th>
      <td>2359.698486</td>
      <td>2044.876587</td>
      <td>2089.619385</td>
      <td>2070.818359</td>
      <td>4.703721e+10</td>
      <td>2070.818359</td>
    </tr>
  </tbody>
</table>
<p>2078 rows × 6 columns</p>
</div>




```python
kama_indicator = KAMAIndicator(close = df["Close"], window = 10, pow1 = 2, pow2 = 30, fillna = False)
df['kama'] = kama_indicator.kama()
ppo_indicator = PercentagePriceOscillator(close = df["Close"], window_slow = 20, window_fast = 10, window_sign = 9, fillna = False)
df['ppo'] = ppo_indicator.ppo()
roc_indicator = ROCIndicator(close = df["Close"], window = 12, fillna = False)
df['roc'] = roc_indicator.roc()
macd_indicator = MACD(close = df["Close"], window_slow = 20, window_fast = 12, window_sign = 9, fillna = False)
df['macd'] = macd_indicator.macd()
rsi_indicator = RSIIndicator(close = df["Close"], window = 14, fillna = False)
df['rsi'] = rsi_indicator.rsi()
aroon_indicator = AroonIndicator(close = df["Close"], window = 20, fillna = False)
df['aroon'] = aroon_indicator.aroon_indicator()
boll_indicator = BollingerBands(close = df["Close"], window = 20, window_dev = 2, fillna = False)
df['boll_mavg'] = boll_indicator.bollinger_mavg()
df.rename(columns = {"Close": "price"}, inplace=True)
prices = df['price'].to_numpy()

df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

df.dropna(inplace=True)
df = df.drop(df.columns[[0, 1, 2, 4, 5]], axis=1)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>kama</th>
      <th>ppo</th>
      <th>roc</th>
      <th>macd</th>
      <th>rsi</th>
      <th>aroon</th>
      <th>boll_mavg</th>
      <th>day_of_month</th>
      <th>day_of_week</th>
      <th>month</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-25</th>
      <td>1.159980</td>
      <td>1.405750</td>
      <td>-13.445239</td>
      <td>-36.539248</td>
      <td>-0.166625</td>
      <td>35.737493</td>
      <td>-10.0</td>
      <td>1.340227</td>
      <td>25</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-08-26</th>
      <td>1.147700</td>
      <td>1.402955</td>
      <td>-13.377194</td>
      <td>-32.044529</td>
      <td>-0.162148</td>
      <td>35.483536</td>
      <td>25.0</td>
      <td>1.259006</td>
      <td>26</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-08-27</th>
      <td>1.191380</td>
      <td>1.398493</td>
      <td>-12.841273</td>
      <td>-23.923552</td>
      <td>-0.153785</td>
      <td>37.193189</td>
      <td>25.0</td>
      <td>1.280909</td>
      <td>27</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-08-28</th>
      <td>1.182550</td>
      <td>1.394309</td>
      <td>-12.305072</td>
      <td>-1.749735</td>
      <td>-0.145647</td>
      <td>36.979851</td>
      <td>20.0</td>
      <td>1.304942</td>
      <td>28</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-08-29</th>
      <td>1.319270</td>
      <td>1.390506</td>
      <td>-10.768333</td>
      <td>21.362409</td>
      <td>-0.129269</td>
      <td>42.481178</td>
      <td>15.0</td>
      <td>1.335483</td>
      <td>29</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-04-13</th>
      <td>2435.104980</td>
      <td>2087.636183</td>
      <td>5.462911</td>
      <td>13.618697</td>
      <td>85.377266</td>
      <td>73.005780</td>
      <td>85.0</td>
      <td>2023.104608</td>
      <td>13</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-04-14</th>
      <td>2519.116211</td>
      <td>2151.219375</td>
      <td>6.256303</td>
      <td>24.190903</td>
      <td>99.025381</td>
      <td>75.243166</td>
      <td>90.0</td>
      <td>2063.918317</td>
      <td>14</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-04-15</th>
      <td>2431.946533</td>
      <td>2174.200615</td>
      <td>6.366490</td>
      <td>16.187475</td>
      <td>102.914171</td>
      <td>68.865192</td>
      <td>90.0</td>
      <td>2099.690912</td>
      <td>15</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-04-16</th>
      <td>2344.895020</td>
      <td>2194.967448</td>
      <td>5.996797</td>
      <td>11.243856</td>
      <td>99.281483</td>
      <td>63.111826</td>
      <td>85.0</td>
      <td>2132.367865</td>
      <td>16</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2021-04-18</th>
      <td>2070.818359</td>
      <td>2194.254329</td>
      <td>4.579605</td>
      <td>-2.245139</td>
      <td>78.982640</td>
      <td>49.180422</td>
      <td>80.0</td>
      <td>2144.924536</td>
      <td>18</td>
      <td>6</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>2059 rows × 11 columns</p>
</div>



Im Folgenden ist nun einmal der gesamt geladene Kursverlauf Etherium zu sehen.


```python
prices = df['price'].to_numpy()
days = range(1, len(prices) + 1)

plt.plot(days, prices, 'blue', label='Price')
plt.title('ETH USD Price')
plt.xlabel('Tage')
plt.ylabel('Preis')
plt.legend()

plt.show()
```


    
![png](output_58_0.png)
    


### Aufteilung in Features und Labels
Die Daten werden in Features und Labels aufgeteilt. In diesem Fall sind die Spalten, welche in `X_columns` definiert sind, die Features und der Preis das Label. 


```python
X_columns = ['price', 'kama', 'ppo', 'roc', 'macd', 'rsi', 'aroon', 'boll_mavg', 
                   'day_of_month', 'day_of_week', 'month']

X_data = df.filter(X_columns)
y_data = df.filter(['price'])

print(X_data.shape)
print(y_data.shape)
```

    (2059, 11)
    (2059, 1)
    

### Transformation der Daten
Für die Transformation der Daten wird der MinMaxScaler verwendet. Dieser skaliert die Daten in einen vorgegebenen Bereich und macht die Daten damit praktikabel. In diesem Fall werden alle Daten in einem Bereich zwischen 0 und 1 skaliert.


```python
X_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))

X_scaled_data = X_scaler.fit_transform(X_data)
y_scaled_data = y_scaler.fit_transform(y_data)

X_scaled_data = pd.DataFrame(data=X_scaled_data, index=X_data.index, columns=X_columns)
y_scaled_data = pd.DataFrame(data=y_scaled_data, index=y_data.index, columns=['price'])

X_scaled_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>kama</th>
      <th>ppo</th>
      <th>roc</th>
      <th>macd</th>
      <th>rsi</th>
      <th>aroon</th>
      <th>boll_mavg</th>
      <th>day_of_month</th>
      <th>day_of_week</th>
      <th>month</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-08-25</th>
      <td>0.000288</td>
      <td>0.000382</td>
      <td>0.087378</td>
      <td>0.076386</td>
      <td>0.333396</td>
      <td>0.257724</td>
      <td>0.447368</td>
      <td>0.000360</td>
      <td>0.800000</td>
      <td>0.166667</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>2015-08-26</th>
      <td>0.000283</td>
      <td>0.000381</td>
      <td>0.089021</td>
      <td>0.097446</td>
      <td>0.333423</td>
      <td>0.254459</td>
      <td>0.631579</td>
      <td>0.000322</td>
      <td>0.833333</td>
      <td>0.333333</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>2015-08-27</th>
      <td>0.000300</td>
      <td>0.000379</td>
      <td>0.101965</td>
      <td>0.135497</td>
      <td>0.333472</td>
      <td>0.276440</td>
      <td>0.631579</td>
      <td>0.000332</td>
      <td>0.866667</td>
      <td>0.500000</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>2015-08-28</th>
      <td>0.000297</td>
      <td>0.000377</td>
      <td>0.114915</td>
      <td>0.239393</td>
      <td>0.333520</td>
      <td>0.273698</td>
      <td>0.605263</td>
      <td>0.000343</td>
      <td>0.900000</td>
      <td>0.666667</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>2015-08-29</th>
      <td>0.000351</td>
      <td>0.000375</td>
      <td>0.152030</td>
      <td>0.347686</td>
      <td>0.333617</td>
      <td>0.344430</td>
      <td>0.578947</td>
      <td>0.000358</td>
      <td>0.933333</td>
      <td>0.833333</td>
      <td>0.636364</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-04-13</th>
      <td>0.966645</td>
      <td>0.951089</td>
      <td>0.544041</td>
      <td>0.311402</td>
      <td>0.838183</td>
      <td>0.736895</td>
      <td>0.947368</td>
      <td>0.943190</td>
      <td>0.400000</td>
      <td>0.166667</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>2021-04-14</th>
      <td>1.000000</td>
      <td>0.980064</td>
      <td>0.563203</td>
      <td>0.360939</td>
      <td>0.918719</td>
      <td>0.765662</td>
      <td>0.973684</td>
      <td>0.962224</td>
      <td>0.433333</td>
      <td>0.333333</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>2021-04-15</th>
      <td>0.965391</td>
      <td>0.990536</td>
      <td>0.565864</td>
      <td>0.323438</td>
      <td>0.941666</td>
      <td>0.683658</td>
      <td>0.973684</td>
      <td>0.978906</td>
      <td>0.466667</td>
      <td>0.500000</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>2021-04-16</th>
      <td>0.930828</td>
      <td>1.000000</td>
      <td>0.556935</td>
      <td>0.300275</td>
      <td>0.920230</td>
      <td>0.609685</td>
      <td>0.947368</td>
      <td>0.994144</td>
      <td>0.500000</td>
      <td>0.666667</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>2021-04-18</th>
      <td>0.822011</td>
      <td>0.999675</td>
      <td>0.522708</td>
      <td>0.237072</td>
      <td>0.800449</td>
      <td>0.430564</td>
      <td>0.921053</td>
      <td>1.000000</td>
      <td>0.566667</td>
      <td>1.000000</td>
      <td>0.272727</td>
    </tr>
  </tbody>
</table>
<p>2059 rows × 11 columns</p>
</div>



Um das Modell trainieren zu können, müssen zunächst die Daten in batches unterteilt werden und danach die batches vermischt werden. 


```python
X_scaled_batches = []
y_scaled_batches = []

for i in range(len(X_scaled_data) - batch_size - 1):
    X_scaled_batches.append(X_scaled_data.iloc[i:(i+batch_size)].values)
    y_scaled_batches.append(y_scaled_data.iloc[i+batch_size + 1])

```


```python
mixed = list(zip(X_scaled_batches, y_scaled_batches))

random.shuffle(mixed)

X_random_batches, y_random_batches = zip(*mixed)
```

Die gesammelten Daten müssen im nächsten Schritt in Traings- und Testdaten aufgeteilt werden. Dafür wurde die Aufteilung von 90% zu 10% gewählt (90% Traningsdaten und 10% Testdaten). Beide Datensätze haben immernoch die gleiche Anzahl an Spalten, die Zeilen wurden entsprechend der genannten Aufteilung gesplittet.


```python
train_size = int(len(X_scaled_batches) * 0.9)
test_size = len(X_scaled_batches) - train_size
X_train_random, X_test_random = X_random_batches[0:train_size], X_random_batches[train_size:len(X_scaled_batches)]
y_train_random, y_test_random = y_random_batches[0:train_size], y_random_batches[train_size:len(y_scaled_batches)]

X_train_random = np.array(X_train_random)
X_train_random = np.reshape(X_train_random, (X_train_random.shape[0], X_train_random.shape[1], X_train_random.shape[2]))
y_train_random = np.array(y_train_random)

X_test_random = np.array(X_test_random)
X_test_random = np.reshape(X_test_random, (X_test_random.shape[0], X_test_random.shape[1], X_test_random.shape[2]))
y_test_random = np.array(y_test_random)

```


```python
X_train, X_test = X_scaled_batches[0:train_size], X_scaled_batches[train_size:len(X_scaled_batches)]
y_train, y_test = y_scaled_batches[0:train_size], y_scaled_batches[train_size:len(y_scaled_batches)]

X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
y_train = np.array(y_train)

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
y_test = np.array(y_test)

X_train_random.shape
```




    (1824, 31, 11)



@michael

### Modell erstellen
Bei diesem Anwendungsbeispiel ist das Ziel den Kursverlauf anhand von Indikatoren und Preisen aus der Vergangenheit für die Zukunft vorherzusagen. Im Detail werden 11 Kennzahlen für jeden der letzten 30 Tage verwendet, um den Preis für morgen vorherzusagen.

Damit das Ziel erreicht werden kann, wird ein LSTM(long short-term memory)-Modell verwendet. Dieses Modell ist eine Erweiterung zu dem RNN(recurrent neural network)-Modell. Das LSTM-Modell im spezielle ist dafür ausgelegt in Zeitreihen oder anderen zusammenhängenden Datensätzen bestimmte Sequenzen zu erkennen. Neugewonnene Informationen können dabei gespeichert werden, um bei zukünftigen Zeitreihen angewandt zu werden. Außerdem kann ein LSTM-Modell entscheiden, ob eine Zeitreihe wichtige Informationen enthält oder nicht und diese dann entweder vergessen oder aktualisieren.
Für das LSTM-Modell werden folgende Parameter definiert: 

* `units = 15` (passende Anzahl für die Menge an Daten; bei höherer Anzahl --> Overfitting)
* `return_sequences = False` (Nur eine LSTM-layer --> False)
* `input_shape = 31, 11` (Diese Zahlen spiegel die Form der Inputdaten wider; 31: batch_size; 11: Anzahl der Indikatoren) 

Anschließend wird für Dropout bestimmt wie viel Prozent der Neuronen pro Durchlauf "ausgeschaltet" sind, um die Gefahr von Overfitting zu vermeiden.

Der letzte Bestandtteilt ist die Dense Layer. Dort wird das Outputformat definiert. Die Anzahl an `units` entspricht in diesem Beispiel 1, da nur der Preis für morgen verhergesagt werden soll. Sollten beispielsweise die Preise für die nächsten 3 Tage vorhergesatz werden, müsste die Dense-layser mit 3 definiert werden.  

In der `model.summary` können nochmal die Daten überprüft werden.


```python
model = tf.keras.Sequential()

model.add(layers.LSTM(units = 15, return_sequences = False, input_shape = (X_train_random.shape[1], X_train_random.shape[2])))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units = 1))

model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 15)                1620      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 15)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 16        
    =================================================================
    Total params: 1,636
    Trainable params: 1,636
    Non-trainable params: 0
    _________________________________________________________________
    

### Modell trainieren
Für das Modell wird zum einen der `adam`-Optimierer und zum anderen die `mean_squared_error` loss-Funktion genutzt.

Für den `adam`-Optimierer haben wir uns, wie bereits oben beschrieben, entschieden, weil sich die Learningrate mit der Zeit automatisch anpasst und somit die Weights verbessert. Die Learningrate wurde hier nicht angepasst und ist damit standardmäßig auf 0,001 eingestellt. Dies ist möglich, da bei diesem Modell nicht so viele Daten zur Verfügung stehen und damit das Problem des Overfittings hier nicht eintritt.

Die loss-Funktion `mean_aquared_error` ist für diesen Anwendungsfall sehr geeignet, weil die es bei diesem Modell darum geht, möglichst nahe an den tatsächlichen Output zu kommen. Mit dieser Funktion wird als Gundidee immer der vorhergesagte Output von dem tatsächlichen Output abzogen und davon das Quadrat benutzt. Damit kann bei diesem LSTM-Modell ein sehr niedriger loss-Wert erreicht werden. 


```python
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train_random, y_train_random,
    epochs=30,
    validation_split=0.1,
)
```

    Epoch 1/30
    52/52 [==============================] - 2s 15ms/step - loss: 0.0830 - val_loss: 0.0049
    Epoch 2/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0120 - val_loss: 0.0022
    Epoch 3/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0069 - val_loss: 0.0016
    Epoch 4/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0055 - val_loss: 0.0014
    Epoch 5/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0043 - val_loss: 0.0012
    Epoch 6/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0037 - val_loss: 0.0012
    Epoch 7/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0031 - val_loss: 0.0011
    Epoch 8/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0030 - val_loss: 9.8658e-04
    Epoch 9/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0028 - val_loss: 0.0010
    Epoch 10/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0023 - val_loss: 8.7019e-04
    Epoch 11/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0027 - val_loss: 8.5181e-04
    Epoch 12/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0022 - val_loss: 8.3948e-04
    Epoch 13/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0018 - val_loss: 8.3702e-04
    Epoch 14/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0021 - val_loss: 7.2040e-04
    Epoch 15/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0017 - val_loss: 7.8280e-04
    Epoch 16/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0017 - val_loss: 6.6268e-04
    Epoch 17/30
    52/52 [==============================] - 1s 10ms/step - loss: 0.0020 - val_loss: 7.6451e-04
    Epoch 18/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0016 - val_loss: 6.7662e-04
    Epoch 19/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0019 - val_loss: 6.6556e-04
    Epoch 20/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0016 - val_loss: 6.3437e-04
    Epoch 21/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0015 - val_loss: 6.1234e-04
    Epoch 22/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0015 - val_loss: 5.6071e-04
    Epoch 23/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0017 - val_loss: 5.4795e-04
    Epoch 24/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0017 - val_loss: 5.4818e-04
    Epoch 25/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0012 - val_loss: 5.8539e-04
    Epoch 26/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0015 - val_loss: 6.3964e-04
    Epoch 27/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0014 - val_loss: 5.4724e-04
    Epoch 28/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0015 - val_loss: 5.6394e-04
    Epoch 29/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0015 - val_loss: 5.6015e-04
    Epoch 30/30
    52/52 [==============================] - 0s 8ms/step - loss: 0.0014 - val_loss: 5.2395e-04
    

Die loss-Rate sollte bei einem Modell immer so gering wie nur möglich sein. In dem folgendem Diagramm ist gut zu sehen, dass die loss-Rate in den ersten Epochen noch relativ hoch war und sich dann immer mehr einer Zahl nahe 0,0015 angegelichen hat. Die Rate wurde dann auch ziemlich konstant über die restlichen Epochen gehalten. 


```python
history_dict = history.history
history_dict.keys()
```




    dict_keys(['loss', 'val_loss'])




```python
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'o', color='orange', label='Training Loss')
plt.plot(epochs, val_loss, 'blue', label='Validation Loss')
plt.title('Trainings- und Validationsloss')
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.legend()

plt.show()
```


    
![png](output_75_0.png)
    


### Überprüfung des Modells
Nachdem das Modell nun trainiert ist, kann zunächst überprüft werden, wie das Modell mit den Trainingsdaten performed. In dem ersten Diagramm sind alle Trainingsdaten abgebildet. Im zweiten Diagramm sind die Vorhersagen des ersten Jahres und im letzten Diagramm die Vorhersagen des letzten Jahres eingezeichnet. Dort ist gut zu erkennen, dass im ersten Jahr die Vorhersage noch sehr ungenau ist und große Schwankungen enhält. Im Gegensatz dazu ist die Vorhersage im letzten Jahr ziemliche nahe am tatsächlichen Kurs.


```python
predicted_price = model.predict(X_train)
predicted_price = y_scaler.inverse_transform(predicted_price)
y_train = y_scaler.inverse_transform(y_train)
```


```python
plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="true")
plt.plot(np.arange(0, len(y_train)), predicted_price, 'r', label="prediction")
plt.ylabel('Price')
plt.xlabel('Time Step')
plt.legend()
plt.show();
```


    
![png](output_78_0.png)
    



```python
plt.plot(np.arange(0, 365), y_train[:365], 'g', label="true")
plt.plot(np.arange(0, 365), predicted_price[:365], 'r', label="prediction")
plt.ylabel('Price')
plt.xlabel('Time Step')
plt.legend()
plt.show();
```


    
![png](output_79_0.png)
    



```python
plt.plot(np.arange(0, 365), y_train[-365:], 'g', label="true")
plt.plot(np.arange(0, 365), predicted_price[-365:], 'r', label="prediction")
plt.ylabel('Price')
plt.xlabel('Time Step')
plt.legend()
plt.show();
```


    
![png](output_80_0.png)
    


### Test des Modells
Nachdem das Modell nun trainiert ist, kann mit den Testdaten überprüft werden, wie gut das Modell funktioniert. Das Diagramm zeigt dabei in blau den tatsächlichen Preisverlauf während der Testphase ab und in rot den vom Modell vorhergesagten Preisverlauf während der Testphase.


```python
predicted_price = model.predict(X_test)
predicted_price = y_scaler.inverse_transform(predicted_price)
y_train_inv = y_scaler.inverse_transform(y_train)
y_test_inv = y_scaler.inverse_transform(y_test)
predicted_price

plt.plot(y_test_inv, label="true")
plt.plot(predicted_price, 'r', label="prediction")
plt.ylabel('Price')
plt.xlabel('Time Step')
plt.legend()
plt.show();
```


    
![png](output_82_0.png)
    


@aaron

## 3. Auführung

## Anwenden auf Twitter Livedaten

Da die Sentimentanalyse lediglich eine Ergänzung zu der technischen Analyse ist müssen die Ergebnisse entsprechend aufbereitet werden.

Um die Tweets zu fetchen wird `searchtweets` verwendet. Weiterhin wird wieder Matplotlib verwendet, um die Ergebnisse graphisch darzustellen.


```python
# pip install searchtweets-v2
from searchtweets import load_credentials, gen_request_parameters, collect_results
from datetime import datetime
import matplotlib.dates as mdate
import math
```

Hier laden wir den Token für die Twitter API, dieser sollte sich im Rootordner des Projekts liegen.


```python
search_args = load_credentials("./.twitter_keys.yaml", yaml_key="search_tweets_v2")
```

Hier definieren wir unsere Queryparameter. Wir laden 100 Tweets, was das Maximum für einen einzelnen API Request ist und geben an, dass alle Tweets mit den Keywords "ether", "eth", "ethereum" oder "cryptocurrency" gefetcht werden sollen. Weiterhin filtern wir Tweets von Bots heraus und Tweets, die das Wort "app" enthalten, da dies meist nur Werbung ist. Zusätzlich müssen die Nutzer verifiziert sein und die Sprache englisch.


```python
max_tweets = 100

query = gen_request_parameters(
    "(ether OR eth OR ethereum OR cryptocurrency) -bot -app -is:retweet is:verified lang:en",
    tweet_fields="id,created_at,text,public_metrics",
    results_per_call=max_tweets)
```

Mit `collect_results()` fetchen wir nun die Tweets und reversen sie, da sie hier für uns falschrum ankommen (neuester Tweet kommt hier zuerst). Mit `pop()` entfernen wir das erste Element, da es sich hier um ein Informationsobjekt handelt. Weiterhin filtern wir die für uns relevanten Informationen heraus wie Datum und Text.


```python
tweets = list(reversed(collect_results(query, max_tweets=max_tweets, result_stream_args=search_args)))

tweets.pop(0)

print(tweets[0])

create_dates = []
tweet_texts = []

for tweet in tweets:
    if 'text' not in tweet:
        continue
    tweet_texts.append(tweet['text'])
    
    utc_time = datetime.strptime(tweet['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()
    create_dates.append(epoch_time)
```

    {'id': '1383624271822417923', 'created_at': '2021-04-18T03:31:47.000Z', 'text': 'This scam claimed to allow clients to invest in a trading pool managed by a “master trader” called Steven Twain. #MTI #bitcoin #cryptocurrency https://t.co/DQ5ochtwfC', 'public_metrics': {'retweet_count': 13, 'reply_count': 4, 'like_count': 44, 'quote_count': 1}}
    

Hier übergeben wir unseren Sentimentmodel den Batch an gefetchten Tweets. Dannach formatieren wir noch das Sentiment, sodass es von -1 bis 1 geht. Somit kann man besser unterscheiden, ob Tweets negativ oder positiv gemeint sind.


```python
raw_sentiment = sentiment_model.predict(tweet_texts).flatten()

sentiment = []

for s in raw_sentiment:
    sentiment.append((s - 0.5) * 2)

for i in range(5):
    d = create_dates[i]
    t = tweet_texts[i]
    p = sentiment[i]
    print("{} - {} - {}".format(d,t,p))
```

    1618716707.0 - This scam claimed to allow clients to invest in a trading pool managed by a “master trader” called Steven Twain. #MTI #bitcoin #cryptocurrency https://t.co/DQ5ochtwfC - 0.3408404588699341
    1618716765.0 - The digital asset #Ethereum touched an all-time high on April 16, 2021, reaching $2,533 per ETH. https://t.co/fgMK0mVgFS - 0.31600141525268555
    1618716962.0 - Tax cheats cost the U.S. far more than previously thought — and cryptocurrency is part of the problem, IRS commissioner says https://t.co/uUwC4dowjr - -0.24571555852890015
    1618717502.0 - After Ethering Serius Jones, ARP Emerges As Most Dangerous Man in Battle Rap https://t.co/hj7Be37OHh - 0.27101635932922363
    1618718031.0 - just made the eth long of my life - 0.2723346948623657
    

Da wir den aktuellen Sentimenttrend bestimmen wollen implementieren wir eine Simple Moving Average


```python
def simple_moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)
```

Nun plotten wir das Sentiment in einen Zeitgraphen. Hier können wir bereits erkennen, wir die aktuelle Marktstimmung der letzten Stunden aussieht


```python
n = 10
sma = simple_moving_avg(sentiment, n)

for i in range(n - 1):
    sma = np.insert(sma, i, sentiment[i])
    
dates = mdate.epoch2num(create_dates)

fig, ax = plt.subplots()

ax.plot(dates, sma, label="simple moving average")
ax.plot(dates, sentiment, "o", label="predicted sentiment")

date_fmt = '%d-%m-%y %H:%M:%S'

date_formatter = mdate.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)

fig.autofmt_xdate()

plt.title('Predictions and MA')
plt.legend(loc='lower right')

plt.show()
```


    
![png](output_97_0.png)
    


Wir wollen aber zusätzlich noch den Einfluss der Tweets miteinberechnen. Dafür gewichten wir die Tweets mithilfe der Anzahl der Likes. Dafür mulitplizieren wir die Anzahl der Likes mit dem Sentiment Wert (+1 da sonst alle Tweets mit 0 Likes eliminiert werden). Mit den Sigmoid Funktion squashen wir alle Werte zurück in unseren vorherigen Wertebereich.


```python
weighted_sentiment = []

def normalized_sigmoid(x):
  return ((1 / (1 + math.exp(-x))) - 0.5) * 2

for i in range(len(sentiment)):
    if 'public_metrics' not in tweets[i]:
        weight = 1
        weighted_sentiment.append(sentiment[i])
    else:
        weight = tweets[i]['public_metrics']['like_count'] + 1
        weighted_sentiment.append(normalized_sigmoid(weight * sentiment[i]))

```

Nun können wir die gewichtete Marktstimmung erneut ausgeben


```python
n = 10
weighted_sma = simple_moving_avg(weighted_sentiment, n)

for i in range(n - 1):
    weighted_sma = np.insert(weighted_sma, i, weighted_sentiment[i])
    
dates = mdate.epoch2num(create_dates)

fig, ax = plt.subplots()

ax.plot(dates, weighted_sma, label="simple moving average")
ax.plot(dates, weighted_sentiment, "o", label="weighted sentiment")

date_fmt = '%d-%m-%y %H:%M:%S'

date_formatter = mdate.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)

fig.autofmt_xdate()

plt.title('Weighted predictions and MA')
plt.legend(loc='lower right')

plt.show()
```


    
![png](output_101_0.png)
    


Letztendlich können wir die beiden Werte noch vergleichen, um zu überprüfen, ob die Gewichtung tatsächlich einen Einfluss auf den Stimmungstrend hat


```python
dates = mdate.epoch2num(create_dates)

fig, ax = plt.subplots()

ax.plot(dates, sma, label="raw", color='blue')
ax.plot(dates, weighted_sma, label="weighted", color='orange')

date_fmt = '%d-%m-%y %H:%M:%S'

date_formatter = mdate.DateFormatter(date_fmt)
ax.xaxis.set_major_formatter(date_formatter)

fig.autofmt_xdate()

plt.title('Raw and weighted predictions')
plt.legend(loc='lower right')

plt.show()
```


    
![png](output_103_0.png)
    


Die folgende Funktion dient zur Datenaufbereitung für die Vorhersage des Kursverlaufes 30 Tage in die Zukunft.


```python
def create_data(df, X_scaler_predict, y_scaler_predict):
    kama_indicator = KAMAIndicator(close = df["price"], window = 10, pow1 = 2, pow2 = 30, fillna = False)
    df['kama'] = kama_indicator.kama()
    ppo_indicator = PercentagePriceOscillator(close = df["price"], window_slow = 20, window_fast = 10, window_sign = 9, fillna = False)
    df['ppo'] = ppo_indicator.ppo()
    roc_indicator = ROCIndicator(close = df["price"], window = 12, fillna = False)
    df['roc'] = roc_indicator.roc()
    macd_indicator = MACD(close = df["price"], window_slow = 20, window_fast = 12, window_sign = 9, fillna = False)
    df['macd'] = macd_indicator.macd()
    rsi_indicator = RSIIndicator(close = df["price"], window = 14, fillna = False)
    df['rsi'] = rsi_indicator.rsi()
    aroon_indicator = AroonIndicator(close = df["price"], window = 20, fillna = False)
    df['aroon'] = aroon_indicator.aroon_indicator()
    boll_indicator = BollingerBands(close = df["price"], window = 20, window_dev = 2, fillna = False)
    df['boll_mavg'] = boll_indicator.bollinger_mavg()
    df['day_of_month'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    prices = df['price'].to_numpy()
    
    increase = []
    for i in range(0, len(df)):
        if(i == len(prices)-1):
            increase.append(1)
        else:
            if(prices[i+1] > prices[i]):
                increase.append(1)
            else:
                increase.append(0)
    df['increase'] = increase

    df_new = df.tail(batch_size)
    predict = df_new.iloc[:len(df_new)]
    

    X_predict = predict.filter(X_columns)
    y_predict = predict.filter(['price'])

    X_scaled_predict = X_scaler_predict.fit_transform(X_predict)
    y_scaled_predict = y_scaler_predict.fit_transform(y_predict)

    X_scaled_predict = pd.DataFrame(data=X_scaled_predict, index=predict.index, columns=X_columns)
    y_scaled_predict = pd.DataFrame(data=y_scaled_predict, index=predict.index, columns=['price'])
    
    return df, X_scaled_predict, y_scaled_predict
```

### Vorhersage des Kurses 30 Tage in die Zukunft
Für die Vorhersage werden die Daten der letzten 81 Tage abgefragt, anschließend werden wieder alle Indicatoren hinzugefügt. Danach wird der Datensatz wieder auf 31 Einträge gekürzt (wie Batch-Größe) und eine Vorhersage für den nächsten Tag gemacht. Anhand von dem vorhergesagten Preis müssen mit der Funktion oben wieder die verschiedenen Indikatoren berechnet und die Daten in das richtige Format gebracht werden. Danach wird wieder eine Vorhersage für den nächsten Tag gemacht und dies wiederholt sich 30 mal und am Ende ist ein Diagramm mit der Vorhersage für die nächsten 30 Tage zu sehen. 


```python
end = datetime.today()
start = datetime.today() - timedelta(days=batch_size + 50)
ETH = pdr.DataReader(symbol,'yahoo',start,end)

df = pd.DataFrame(data=ETH)
df =  df.drop(df.columns[[0, 1, 2, 4, 5]], axis=1)
df.rename(columns = {"Close": "price"}, inplace=True)
prices = df['price'].to_numpy()

X_scaler_predict = MinMaxScaler(feature_range = (0, 1))
y_scaler_predict = MinMaxScaler(feature_range = (0, 1))

days_in_future = 30
y_predicted_all = []

for i in range(days_in_future):
    df, X_scaled_predict, y_scaled_predict = create_data(df, X_scaler_predict, y_scaler_predict)
    X = np.array([X_scaled_predict.values])
    y_predicted = model.predict(X)
    y_predicted_inv = y_scaler_predict.inverse_transform(y_predicted)
    y_predicted_all.append(y_predicted_inv[0][0])
    del X_scaled_predict
    del y_scaled_predict
    
    add_index = pd.Index([(datetime.today())+ timedelta(days=1)])
    add_index.set_names('Date', inplace=True)
    df2 = pd.DataFrame(index = add_index, data=([[y_predicted_inv[0][0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0]]), columns=X_columns)
    df = df.append(df2)
    del y_predicted
    del y_predicted_inv

print(y_predicted_all)
```

    [2284.8516, 2308.7874, 2323.3335, 2334.8848, 2351.8909, 2365.3315, 2377.4165, 2392.6086, 2404.4707, 2410.1423, 2404.9104, 2405.7192, 2411.0747, 2406.7961, 2378.739, 2369.7656, 2368.461, 2367.797, 2372.501, 2380.9988, 2404.8696, 2410.6248, 2416.4885, 2420.61, 2426.1545, 2429.0803, 2428.9766, 2427.2515, 2383.3, 2380.743]
    


```python
plt.plot(np.arange(0, len(prices)), prices, 'g', label="history")
plt.plot(np.arange(len(prices) - 1, len(prices) + len(y_predicted_all)), [prices[-1]] + y_predicted_all, 'r', label="prediction")
plt.title('ETH USD Price with prediction')
plt.xlabel('Tage')
plt.ylabel('Preis')
plt.legend()

plt.show()
```


    
![png](output_108_0.png)
    


#### Herausforderungen bei einem produktiven Einsatz der Modelle

Nachdem die ersten Vorhersagen mit einem Modell getroffen wurde, stellt sich auch die Frage, wie diese Modelle nun innerhalb einer produktiven Umgebung eingesetzt werden können.
Dabei stellen sich einige Herausforderungen:

1. **Wie werden die Modelle bereitgestellt?:**
Es muss eine API programmiert werden, die entsprechende Endpunkte bereitstellt, um mit den Modellen zu interagieren.

2. **Welche Anforderungen können an die API gestellt werden?:**
Es gilt weiterhin zu untersuchen, ob und inwiefern Parameter bei der bisherigen Nutzung des Modells *hardgecodet* wurden, die nun extrahiert und konfigurierbar gemacht werden müssen, sodass eine nutzerindividuelle Anfrage möglich ist.

3. **Datentransformation:**
Die Datentransformation, die während des Trainings und den initialen Vorhersagen durchgeführt wurde muss auch entsprechend der Live-Datenquelle durchgeführt werden. Das bedeutet, evtl. Aufwändige Datenebereinigung oder Vorbereitungen für das Modell müssen innerhalb der API durchgeführt werden. So stellen sich evtl. neue Anforderungen an das Modell, damit Transformationen nicht zu aufwändig sind, um eine Responsivität der API zu gewährleisten.

Zusammenfassend lässt sich sagen, dass es einige Aufgaben zu erledigen gilt, bevor ein trainiertes ML-Modell produktiv genutzt werden kann.
