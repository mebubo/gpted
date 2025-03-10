---
title: Gpted
emoji: 🏃
colorFrom: pink
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# GPTed blog post part 1

This post describes my attempt to build an improved version of GPTed from https://vgel.me/posts/gpted-launch/ and what I learned from it.

## What has been done in the original GPTed

- Use logprobs returned by the OpenAI API (in particular, the [legacy /v1/completions API](https://platform.openai.com/docs/api-reference/completions)) for tokens _in the existing text_ (as opposed to generated text) to detect the tokens the model is surprised by
- Provide a basic text editing UI that has a mode in which the tokens with a logprob below a given threshold are highlighted. Not all highlighted tokens are necessarily a mistake, but the idea is that it may be worth checking that a low-probability token is indeed intended.

## Improvements that I wanted to make

- Operate at the word level, instead of token level, to compute the logprobs of whole words even if they are mutli-token, and to highlight whole words
- Propose replacement words for the highlighted words
	- Specifically, words with probability higher than the flagging threshold

## Walkthorough of the app in action, on a concrete example

![](img/GPTed.jpeg)

TODO

## Implementation details

### Overview

TODO A diagram

### On logprobs in OpenAI API

The original GPTed project relied on the 2 features in the [legacy OpenAI /v1/completions API](https://platform.openai.com/docs/api-reference/completions):

> logprobs: Include the log probabilities on the `logprobs` most likely output tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response. The maximum value for `logprobs` is 5.

> echo: Echo back the prompt in addition to the completion

The echo parameter doesn't exist anymore in the [modern /v1/chat/completions API](https://platform.openai.com/docs/api-reference/chat), making it impossible to get logprobs for an existing text (as opposed to generated text). The legacy completions API is [not available](https://platform.openai.com/docs/models#model-endpoint-compatibility) for modern models like GPT4.

Also, the maximum of 5 for the number of logprobs is also quite limiting: there may well be more than 5 tokens above the threshold, and I would like to be able to take all of them into account.

Moreover, the case of multi-token words meant that it would be convenient to use batching, which is not available over the OpenAI API (there is a [batch API](https://platform.openai.com/docs/guides/batch) but it is not for interactive use).

For the above 3 reasons, I decided to switch to using local models.

### Local models with huggingface transformers

To run inference locally and get the logits I used huggingface transformers. As model, I used Llama 3.2 1B, because it runs fast enough on a CPU to enable local development on my laptop.
The basic usage to get logits for every token in an input is straightforward. First, we get instances os the model and the tokenizer:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "unsloth/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

Next, we can tokenize the text:

```python
input_text = "Hello world!"
inputs = tokenizer(input_text, return_tensors="pt")
```

The tokenizer is of type [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast). The `inputs` returned by the tokenizer contain 1 extra token at the start: it's the "beginning-of-sequence", whether it is added or not is controlled by the [add_special_tokens parameter](https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__.add_special_tokens), `True` by default.


Finally, we can call the model on the results returned by the tokenizer:

```python

with torch.no_grad():
	outputs = model(**inputs)

logits = outputs.logits # Shape: [batch_size, sequence_length, vocab_size]
```

The output of the model is of type [CausalLMOutputWithPast](https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast), and we are using its [logits field](https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast.logits):

> logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

### Computing the logprob for every token in the input:

```python
def calculate_log_probabilities(model: PreTrainedModel, tokenizer: Tokenizer, inputs: BatchEncoding) -> list[tuple[int, float]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # B x T x V
    logits: torch.Tensor = outputs.logits[:, :-1, :]
    # B x T x V
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    # T - 1
    tokens: torch.Tensor = input_ids[0][1:]
    # T - 1
    token_log_probs: torch.Tensor = log_probs[0, range(log_probs.shape[1]), tokens]
    return list(zip(tokens.tolist(), token_log_probs.tolist()))
```

Explanation:
- we drop the logits for the last token, because they correspond to the probability of the next token (we have no use for it because we are not generating text)
- we compute the softmax over the last dimension (vocab size), to obtain the probability distribution over all tokens
- we drop the first token because it is a beginning-of-sequence token
- `log_probs[0, range(log_probs.shape[1]), tokens]` indexes into log_probs such as to extract
  - at position 0 (probability distribution for the first token after the start-of-sequence token) - the logprob value corresponding to the actual first token
  - at position 1 (probability distribution for the second token after the start-of-sequence token) - the logprob value corresponding to the actual second token
  - etc.

TODO: Maybe improve the explanation using diagrams?

![](img/logits-1.png)

![](img/logits-2.png)


### Combining tokens into words

I wrote a very generic `combine` function, that takes a list of values and a function that tells it how to combine two adjacent values into a single value. If the function returns `None`, the values are not combined.

Thanks to the fact that it is generic, it is very easy to test:


```python
def test_add_if_even():
    def add_if_even(x: int, y: int) -> int | None:
        if (x + y) % 2 == 0:
            return x + y
        return None

    assert combine([1, 3, 1, 4], add_if_even) == [4, 1, 4]
    assert combine([1, 3, 2, 4], add_if_even) == [10]
```

Applying this function to the problem of combining tokens into words is just a matter of writing the correct `combine_fn`:

```python
@dataclass
class Tok:
    index: int
    ids: list[int]
    str: str
    logprob: float

def is_beginning_of_word(s: str) -> bool:
    return (s[0] == " " and s[1:].isalpha()) or s.isalpha()

def is_continuation_of_word(s: str) -> bool:
    return s.isalpha()

def merge_tokens(a: Tok, b: Tok) -> Tok | None:
    if is_beginning_of_word(a.str) and is_continuation_of_word(b.str):
        return Tok(a.index, a.ids + b.ids, a.str + b.str, a.logprob + b.logprob)
    return None
```

This handles nicely the computation of combined logprob for words, and allows me to highlight whole words based on a threshold.

### Producing suggestions for replacement words

Here is how I do it:

Extract the contexts (lists of token prefixes -- all tokens up to the word in question) for each flagged word:

```python
contexts = [word.context for _, word in low_prob_words]
```

Create a `Series` for each context (a series has a budget), and bundle them into a `Batch`:
```python
series = []
for i, x in enumerate(contexts):
    series.append(Series(id=i, tokens=x, budget=5.0))

batch = Batch(items=series)
```

Stopping criterion decides when to stop expanding a series

```python
stopping_criterion = create_stopping_criterion_llm(tokenizer)
```

In my case, I stop when the budget is exhausted, and I also stop if the expansion reached a word boundary (I'm only interested in single-word replacements).

Given the batch and the stopping criterion, we can call the expander:

```python
expander = LLMBatchExpander(model, tokenizer)
expanded = expand(batch, expander, stopping_criterion)
```

TODO: Need a much better explanation of the expansion logic. Among other things, explain why we need to do it iteratively. Probably with diagrams. Explain why we use batches.

### Testing the `expand` logic

The `expand` logic is the most complex part of the project, and in order to make it testable, I made it generic, with only a small part that is llm-specific.

Here is what the tests look like:

```python
def test_expander_zero_budget():
    s = Series(id=0, tokens=[1], budget=0.0)
    expanded = expander.expand(Batch(items=[s]))
    expected = ExpansionOneResultBatch(
        items=[ExpansionOneResult(series=s, expansions=[
            Expansion(token=21, cost=-1.0),
            Expansion(token=22, cost=-1.0),
        ])]
    )
    assert expected == expanded
```

They are based on a non-llm expander based on a hardcoded list of possible expansions, so they are very easy to write, straightforward to interpret, and run very fast.

### Frontend and the http backend

TODO A few words about how those are implemented.

### Deployment on huggingface spaces

It was easy to deploy on Huggingface Spaces. All I needed was to create the space in the Huggigface UI, create this [Dockerfile](./Dockerfile), and a metadata section at the beginning of the [README](./README.md) (the `sdk: docker` is what tells it to use the Dockerfile), add the space as a git remote to my repo, and push to it.

https://huggingface.co/spaces/mebubo/gpted

### Performance improvements

- Filtering out low-probability tokens to avoid looping over arrays of vocab_size in pure Python
- Breaking batches into chunks to avoid CUDA OOM

TODO explain


## Discussion

### Comparison with the original GPTed

### Limitations

- Performance on longer inputs

### Limitations of the decoder-only approach

The main limitation of using decoder-only models like GPT or Llama for this task is the unidirectional attention. It means that we are not using the context to the right of the word. This is especially problematic at the start of the text: the first tokens get very little context, so the the probabilities we get from the model are not very useful. The obvious solution is to use a model with bi-directional attention, such as BERT. This comes with its own set of challenges and will be covered in the part 2 of the post.

### Using Cursor during the implementation

### Other potential possibilities / ideas
- Instead of using a local model, investigate using an API of a provider that exposes logprobs e.g. replicate


## Conclusion

TODO
