# AnyBURL

AnyBURL23 comparison to older version
<img width="1618" height="222" alt="image" src="https://github.com/user-attachments/assets/f15c6a17-deaf-486a-895d-1813d26ab20d" />




## Using AnyBURL

## Training the Model
```bash
java -Xmx12G -cp AnyBURL-23-1x.jar de.unima.ki.anyburl.Learn config-learn.properties 
```

## Testing the Model
```bash
java -Xmx12G -cp AnyBURL-23-1x.jar de.unima.ki.anyburl.Apply config-apply.properties
```

## Evaluating the Model
```bash
java -Xmx12G -cp ./AnyBURL-23-1x.jar de.unima.ki.anyburl.Eval config-eval.properties
```

