# resources-spam-filtering

This is a collection of resources for automatic spam filtering. 

It currently contains some dataset loading utility functions and some sample `scikit-learn`      `Pipeline`s to play with. 

Example use:

```python
df = fetch_spambase()
    p = create_pipeline_spambase()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values.astype('int').ravel() 
    scores = cross_val_score(p, X, y, cv=10)   
    print(scores)s
```