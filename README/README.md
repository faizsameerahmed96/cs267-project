# RAG on research papers


## Parsing the documents
After investigating pymupdf,pymupdf4llm and unstructured, we found that using unstructured was the way to go because:
- It provides an easy way to parse a PDF along with extracting images.
- It provides a clean API and is widely used.

We notice that the library cleanly separates each `.category` into some of the following:
```python
    {'FigureCaption',
    'Formula',
    'Header',
    'Image',
    'ListItem',
    'NarrativeText',
    'Table',
    'Title'}
```

Therefore, we can extract sections using a title followed by narrative texts until the next title OR end of document.
![alt text](image.png)
![alt text](image-2.png)


We can also extract images using
![alt text](image-1.png)

## TODO
- [x] Parse the documents and extract text
- [ ] Investigate different chunking strategies
- [ ] Dense retriever vs sparse retriever
- [ ] Check base performance with Milvus


## Installation
- Install all required deps using `pip install -r requirements.txt`