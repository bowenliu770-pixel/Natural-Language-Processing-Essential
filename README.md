# Part B Learning Log: Improving the Emotion Classifier

## 1. What I was trying to improve

In Part A, I used a normal NLP workflow to classify text into 6 emotions.

The 6 emotions were:

- anger
- fear
- joy
- love
- sadness
- surprise

In Part A, I used **TF-IDF + Dense Neural Network**.
This is a good baseline, but TF-IDF mainly looks at word frequency.
It does not understand word meaning very well, and it does not keep word order.

In Part B, I tried to make the model better by using more advanced NLP methods.

My main goal was:

- make the model understand word meaning better
- make the model use sentence order and context
- reduce overfitting
- compare Part B with the Part A baseline
- improve the final test accuracy

The Part B model used **pretrained GloVe word embeddings + Bidirectional LSTM**.


```text
Part A Test Accuracy: 0.8275
Part B Test Accuracy: 0.9190
```

---

# Technique 1: Pretrained GloVe Word Embeddings

## The Concept

Pretrained word embeddings means using word vectors that were already trained before.

In my notebook, I used **GloVe 6B 100d** embeddings.
Each word is changed into a vector of 100 numbers.

These vectors are useful because they can store word meaning.
For example, words like **happy**, **joy**, and **excited** should be closer in meaning than unrelated words.

This is different from TF-IDF.
TF-IDF mainly counts how important a word is in the dataset, but it does not really understand meaning.

## Why I used it

Emotion classification depends a lot on word meaning.

For example:

- "I am happy" is related to joy
- "I am scared" is related to fear
- "I love this" is related to love

GloVe helps because it already learned many word relationships from a large text dataset.
This gives the model better starting knowledge than learning everything from zero.

## Implementation

```python
embedding_dim = 100
possible_glove_paths = [
    "data/glove.6B.100d.txt",
    "glove.6B.100d.txt",
    "embeddings/glove.6B.100d.txt"
]

embeddings_index = {}

with open(glove_path, encoding="utf-8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = vector
```

Then I created an embedding matrix for my dataset words:

```python
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))
matched_words = 0

for word, i in word_index.items():
    if i >= max_words:
        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        matched_words += 1
```

## The Learning

GloVe works by learning from word co-occurrence.
This means it learns which words often appear near each other in real text.

Words used in similar contexts get similar vectors.
This helps the model understand semantic meaning.

For emotion classification, this is useful because different words can show similar feelings.
For example, **happy**, **glad**, and **excited** may all point to positive emotion.

## Impact on Performance

GloVe helped the model use word meaning instead of only word frequency.
This should help Part B perform better than the Part A TF-IDF model, especially when different words have similar emotion meaning.

---

# Technique 2: Light Text Cleaning for Pretrained Embeddings

## The Concept

For Part A, heavy cleaning can be useful.
For example, removing stopwords and using stemming can reduce noise.

But for pretrained embeddings, heavy cleaning can cause problems.

For example, stemming may change:

```text
happy -> happi
```

The word **happi** may not exist in GloVe.
So the model may lose useful word meaning.

## Why I used it

For GloVe, I used lighter cleaning.
This keeps the words closer to their original form, so more words can match the pretrained GloVe vocabulary.

I also kept stopwords because LSTM models use word order.
Words like **not**, **very**, and **so** can change the emotion of a sentence.

## Implementation

```python
def light_clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#[a-zA-Z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["glove_text"] = train_df["text"].apply(light_clean_text)
val_df["glove_text"] = val_df["text"].apply(light_clean_text)
test_df["glove_text"] = test_df["text"].apply(light_clean_text)
```

## The Learning

I learned that preprocessing is not always the same for every model.

For TF-IDF, stemming and stopword removal may be okay.
For pretrained embeddings, keeping real words is more important because the model needs to match them with the pretrained vocabulary.

## Impact on Performance

Light cleaning helped the model keep more useful words.
This should improve GloVe matching and help the model understand sentence meaning better.

---

# Technique 3: Tokenization and Padding

## The Concept

Deep learning models cannot read raw text directly.
The text must be changed into numbers first.

For Part B, I used a **Tokenizer**.
The tokenizer gives each word an index number.

For example:

```text
"i feel happy" -> [5, 20, 43]
```

Then I used padding so every sentence has the same length.
In my notebook, the maximum sentence length was **60 words**.

## Why I used it

LSTM models need input in sequence form.
The tokenizer keeps the word order, which is important for sentence meaning.

Padding is needed because neural networks usually train in batches, and each input in a batch must have the same shape.

## Implementation

```python
max_words = 15000
max_len = 60

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<OOV>"
)

# fit only on training text to avoid data leakage
tokenizer.fit_on_texts(train_df["glove_text"])

X_train_seq = tokenizer.texts_to_sequences(train_df["glove_text"])
X_val_seq = tokenizer.texts_to_sequences(val_df["glove_text"])
X_test_seq = tokenizer.texts_to_sequences(test_df["glove_text"])

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")
```

## The Learning

Tokenization changes text into word numbers.
Padding makes every sentence the same length.

I also learned that the tokenizer must be fitted only on the training data.
This prevents data leakage because the model should not learn vocabulary from validation or test data before evaluation.

## Impact on Performance

Tokenization and padding allowed the model to use sentence order.
This is better for LSTM compared to TF-IDF because TF-IDF loses the original word order.

---

# Technique 4: Bidirectional LSTM

## The Concept

LSTM stands for **Long Short-Term Memory**.
It is a type of neural network designed for sequence data, such as sentences.

A normal LSTM reads a sentence from left to right.
A **Bidirectional LSTM** reads the sentence in both directions:

- forward direction
- backward direction

This gives the model more context.

## Why I used it

Emotion depends on sentence context.

For example:

```text
I thought I would be happy, but I feel sad.
```

If the model only focuses on one word, it may choose the wrong emotion.
The BiLSTM helps because it reads the full sentence and uses word order.

## Implementation

```python
glove_model = Sequential([
    Input(shape=(max_len,)),

    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        name="pretrained_glove_embedding"
    ),

    SpatialDropout1D(0.3),

    Bidirectional(LSTM(
        64,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )),

    GlobalMaxPooling1D(),

    Dense(64, activation="relu"),
    Dropout(0.4),

    Dense(num_classes, activation="softmax")
])
```

## The Learning

The BiLSTM reads each sentence as a sequence.
It can remember important words and connect them with nearby words.

The bidirectional part makes it stronger because it can understand words using both earlier and later context.

For emotion classification, this is useful because one word alone may not be enough.
The whole sentence matters.

## Impact on Performance

BiLSTM should improve the model compared to a basic Dense model because it can use sentence order and context.
This is especially useful for emotions where the meaning depends on the full sentence.

---

# Technique 5: SpatialDropout1D and Dropout

## The Concept

Dropout randomly turns off some parts of the model during training.
This helps stop the model from memorising the training data too much.

**SpatialDropout1D** is similar, but it is used for sequence and embedding data.
It drops whole embedding features instead of only random single values.

## Why I used it

Text models can overfit if they depend too much on certain words.

For example, the model may always connect one word with one emotion, even when the sentence context is different.

Dropout helps the model learn more general patterns.

## Implementation

```python
SpatialDropout1D(0.3)

Bidirectional(LSTM(
    64,
    return_sequences=True,
    dropout=0.2,
    recurrent_dropout=0.2
))

Dense(64, activation="relu")
Dropout(0.4)
```

## The Learning

Dropout makes the model practise without always using the same features.
This makes the model more stable and less likely to overfit.

SpatialDropout1D is useful after the embedding layer because it helps the model not depend too much on only a few embedding dimensions.

## Impact on Performance

Dropout helped reduce overfitting.
It made the Part B model more reliable when testing on unseen data.

---

# Technique 6: Training Callbacks

## The Concept

Callbacks control training automatically.

I used:

- `EarlyStopping`
- `ReduceLROnPlateau`

## Why I used it

Training for too long can cause overfitting.
Also, sometimes the model stops improving because the learning rate is too high.

Callbacks help fix these problems.

## Implementation

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

glove_history = glove_model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

## The Learning

`EarlyStopping` stops training when validation loss does not improve.
It also restores the best weights, so the final model is not just the last model.

`ReduceLROnPlateau` lowers the learning rate when the model gets stuck.
This helps the model make smaller and better updates.

## Impact on Performance

Callbacks helped make training safer.
They also stopped the model from wasting time and reduced the chance of overfitting.

---

# Technique 7: Evaluation and Model Comparison

## The Concept

After training, I evaluated the Part B model using:

- classification report
- confusion matrix
- comparison table
- accuracy bar chart

This helps show whether Part B actually improved the model.

## Implementation

```python
glove_pred_prob = glove_model.predict(X_test_pad)
glove_y_pred = np.argmax(glove_pred_prob, axis=1)
glove_y_true = np.argmax(y_test, axis=1)

print(classification_report(
    glove_y_true,
    glove_y_pred,
    target_names=emotion_order
))
```

Then I compared Part A and Part B:

```python
part_a_test_loss, part_a_test_acc = model.evaluate(X_test, y_test, verbose=0)
part_b_test_loss, part_b_test_acc = glove_model.evaluate(X_test_pad, y_test, verbose=0)

comparison_df = pd.DataFrame({
    "Model": ["Part A: TF-IDF + Dense NN", "Part B: GloVe + BiLSTM"],
    "Test Loss": [part_a_test_loss, part_b_test_loss],
    "Test Accuracy": [part_a_test_acc, part_b_test_acc]
})
```

## The Learning

Accuracy alone is not always enough.
The classification report shows precision, recall, and F1-score for each emotion.

The confusion matrix shows which emotions the model confuses.
For example, the model may confuse **fear** and **sadness** if the texts have similar negative words.

## Impact on Performance

The comparison table shows whether Part B improved over Part A.
If Part B accuracy is higher, it means GloVe and BiLSTM helped the model understand text better.

If Part B accuracy is not higher, the method is still useful because it shows that pretrained embeddings need correct tuning, enough training data, and good hyperparameters.

---

# Final Result

```text
Part A Model: TF-IDF + Dense Neural Network
Part A Test Loss: 0.748065	
Part A Test Accuracy: 0.8275

Part B Model: GloVe + BiLSTM
Part B Test Loss: 0.190722	
Part B Test Accuracy: 0.9190
```

The Part B model is more advanced because it uses word meaning and sentence order.
The Part A model is simpler because it uses TF-IDF and Dense layers.

---

# What I Learned Overall

From Part B, I learned that improving an NLP model is not only about adding more layers.
The type of text representation is very important.

I learned that:

- TF-IDF is useful as a baseline, but it loses word order.
- GloVe embeddings help the model understand word meaning.
- Light cleaning is better for pretrained embeddings because it keeps real words.
- Tokenization and padding change text into a format the LSTM can use.
- BiLSTM can read sentence context in both directions.
- Dropout helps reduce overfitting.
- Callbacks make training safer and more efficient.
- A comparison table is important because it proves whether Part B improved the model.

The most useful method was **GloVe + BiLSTM**, because it allowed the model to use both semantic meaning and sentence order.

---

# Short Presentation Version

For Part B, I used **pretrained GloVe word embeddings** with a **Bidirectional LSTM** model.

In Part A, I used TF-IDF with a Dense Neural Network.
TF-IDF is useful, but it mainly focuses on word frequency and does not understand word order.

GloVe is better because it changes words into dense vectors that store meaning.
For example, words like **happy**, **joy**, and **excited** can have similar vector meanings.

I also used a BiLSTM because it reads the sentence from both directions.
This helps the model understand full sentence context.

I used light text cleaning so the words could still match the GloVe vocabulary.
I also used Dropout, SpatialDropout1D, EarlyStopping, and ReduceLROnPlateau to reduce overfitting and make training safer.

After running the notebook, I compared Part A and Part B using the test accuracy, classification report, confusion matrix, and comparison bar chart.

This Part B method matches the project brief because it explores **Embedding Strategies** and **Advanced NLP Architectures**.
