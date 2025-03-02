## Part 1

What I want to cover:
- [ ] The original blog post
-  [ ]Improvements that I wanted to make:
	- [ ] In addition to highlighting low-probability words, show replacement suggestions that are more likely
	- [ ] Operate at the level of whole words, not tokens
- [ ] Justification for using a local model
	- [ ] Limitations of the logprobs returned by the APIs
- [ ] Main parts of the project
	- [ ] Combining tokens into words to get the probabilities of whole words
	- [ ] The batched multi-token expansion with probability budget
	- [ ] Testable abstract implementation

## A digression on encoder vs decoder, unidirectional vs bidirectional attention, and whether we could use bidirectional attention for text generation

It is a common misconseption that autoregressive text generation _requires_ unidirectional attention, whereas in reality it is only a matter of efficiency (efficiency at both training and inference time). It is possible to train models with bidirectional attention on next token prediction, and to use them autoregressively at inference, and arguably it would give better quality than unidirectional attention (the bidirectional flow of information between tokens in the current prefix can only be beneficial, e.g. if we are generating the next token in "the quick brown fox jumped over", there is no benefit in not letting "fox" to see "jumped"). However, bidirectional attention would mean that we cannot learn from every token in a text by passing only 1 instance of it through the model, we would have to pass every prefix individually. And at inference time, it would rule out the techniques such as KV caches which are used ubiquitously at all modern LLM deployments for inference, because all attention would need to be recomputed for every prefix.

## Part 2

Applying encoder-only models (those with bidirectional attention) to this task presents several challenges.

Whereas unidierctional attention in decoder-only models enables them to be efficiently trained on the task of next token predition, and used for autoregressive text generation, with an important property of returning logprobs for every input token as a byporduct, encoder-only only models like BERT are trained on masked token prediction (also on next sentence prediction), and it is from this fact that the difficulties arise:

- We cannot get logprobs for all tokens in a given text by passing a single instance of it through the modesl. Instead, because we need to mask individual tokens, replicating the input as many times as there are tokens. It can still be done in 1 pass / 1 batch, but the size of the batch in N_tokens instead of 1 in the decoder-only case.
- For multi-token words, it is not clear if replacing them by a sequence of mask tokens would give results (if model is trained to predict multiple adjacent mask tokens)
- Generating replacesments poses an additional difficulty: we don't know beforehand how many tokens the replacement word would consist of, so naively we'd need to try all possible sequences [MASK], [MASK][MASK], [MASK][MASK][MASK], and so on until a reasonable limit of the number of tokens in a word.

Even if we get the logprobs for a sequence of mask tokens, how do we interpret them? What we need in order to generate candidate words (and to compute their probability) are _conditional_ probabilities of the second token given the first one, the third one given the first two, and so on, but logporbs for a sequence of mask tokens don't give us that.

Speculation: either the logprobs of the second [MASK] in a sequence represent probabilities of tokens at that place _given that the previous token is [MASK]_ (and of course given all other actual non-mask and mask tokens at all other positions), or they represent probabilities for tokens in the second position averaged over all possible tokens in position 1, possibly roughly weighted according to the probabilities of the tokens at position 1 (FIXME: is there even a way to know this?)

**The Cauchy-Schwarz Inequality**

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
