# Feature Space Tools
For now, a feature space explorer written in Python using `transformers` and `matplotlib`. Allows you to browse embeddings in a 3D plot. You can zoom in and out, dump values to a `pandas`-compatible format, &c.

## Usage
1. Edit `src/main.py` to pick preferred Transformer model (e.g., `gpt2`, `bert-base-uncased`) and `device` (e.g., `cpu`)
2. Pick 2+ ascii documents to compare. Run `python src/main.py $text1 $text2 $text3…`
3. Explore!

## Notes
The program will cache sentence embeddings for all sentences in an English-written program in JSON files within `cache`. Each sentence has four keys, one for the sentence and three for the `x`, `y`, `z` coordinates for the sentence’s respective vector. The program will repack the latter three columns to display vectors.

## Example
![example](https://git.sr.ht/~srhm/feature-space-tools/tree/master/item/example.png)

The above image provides a sense of what one should expect to see with this tool. In it, two books have had their sentences plotted out with GPT-2: one work of fiction and one work of non-fiction. While most of the sentences converge on each other, a clear wedge of sentence appears to extend southwards. Re-examining this collection of sentences in both work reveal them to be chapter labels and page numbers, structural strings whose semantic quality is different than the average sentence in a book. Perhaps not surprising, but certainly nice to see visually.
