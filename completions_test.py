from completions import calculate_log_probabilities, load_model, tokenize, split_into_words

model, tokenizer, device = load_model()

def test_text_to_words():
    text = """Hello
world!"""
    token_probs = calculate_log_probabilities(model, tokenizer, tokenize(text, tokenizer, device))
    words = split_into_words(token_probs, tokenizer)
    expected_words = ["Hello", "\n", "world", "!"]
    assert [w.text for w in words] == expected_words

def test_multiline():
    text = """// Context: C code from an image manipulation library.
for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
        buf[y * HEIGHT + x] = 0;
    }
}"""
    tokenized = tokenize(text, tokenizer, device)
    print(tokenized)
    token_probs = calculate_log_probabilities(model, tokenizer, tokenized)
    words = split_into_words(token_probs, tokenizer)
    print("---", [w.text for w in words])
    expected_words = ["//", " Context", ":", " C", " code", " from", " an", " image", " manipulation", " library", ".\n",
                      "for", "(", "int", "y", "=", "0", ";", "y", "<", "HEIGHT", ";", "y", "+", "+", ")", "{", "\n", "    ", "for", "(", "int", "x", "=", "0", ";", "x", "<", "WIDTH", ";", "x", "+", "+", ")", "{", "\n", "        ", "buf", "[", "y", "*", "HEIGHT", "+", "x", "]", "=", "0", ";", "\n", "    ", "}", "\n", "}"]
    assert [w.text for w in words] == expected_words
