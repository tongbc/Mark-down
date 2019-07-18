#  JigSaw比赛

non-trivial loss function.

I tried the custom loss function.
for 1 mil CV 0.9368 LB 0.9376
for 1.2 mil CV 0.9356 LB 0.9387
for 1.26 mil CV 0.9385 LB 0.9380
Any idea what am i doing wrong? I am doing stratified sampling for train and val i.e after selecting random rows as per your kernel.
Also i am using cosine lr schedule? Any harm using that? Should i switch to linear lr schedule?



Yes, to make it work just change model and tokenizer. 
Also you can swap the optimizer (from BERTAdam to OpenAIAdam):

```
from pytorch_pretrained_bert import OpenAIAdam

num_train_optimization_steps = int(EPOCHS*len(train_dataset)/batch_size/accumulation_steps)

optimizer = OpenAIAdam(model.parameters(), 
                       lr=lr,
                       warmup=0.05,
                       t_total=num_train_optimization_steps)
```

Probably, it's some kind more natural and сan help convergence

