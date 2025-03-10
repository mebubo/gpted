import React, { useState } from "react"
import { WordChip } from "./WordChip"
import { Spinner } from "./Spinner"
import { Word } from "../interfaces"

async function checkText(text: string): Promise<Word[]> {
  const encodedText = encodeURIComponent(text);
  const response = await fetch(`/check?text=${encodedText}`);
  const data = await response.json();
  console.log(data);
  return data.words;
}

export default function App() {
  const [threshold, setThreshold] = useState(-5.0)
  const [context, setContext] = useState("")
  const [wordlist, setWordlist] = useState("")
  const [showWholePrompt, setShowWholePrompt] = useState(false)
  // const [text, setText] = useState("I just drove to the store to but eggs, but they had some.")
  const [text, setText] = useState("I drove to the stove to but eggs")
  const [mode, setMode] = useState<"edit" | "check">("edit")
  const [words, setWords] = useState<Word[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const check = async (text: string) => {
    setIsLoading(true)
    try {
      const checkedWords = await checkText(text)
      setWords(checkedWords)
    } finally {
      setIsLoading(false)
      setMode("check")
    }
  }

  const toggleMode = async () => {
    if (mode === "edit") {
      setIsLoading(true)
      await check(text)
    } else {
      setMode("edit")
    }
  }

  const handleReplace = async (index: number, newWord: string) => {
    console.log("handleReplace", index, newWord)
    const updatedWords = words.map((w, i) => {
      if (i === index) {
        return { text: newWord, logprob: 0, replacements: [] }
      }
      return w
    })
    setWords(updatedWords)
    const newText = updatedWords.map(w => w.text).join("")
    setText(newText)
    await check(newText)
  }

  let result

  if (mode === "edit") {
    result = (
      <div className="result-container">
        {isLoading && <Spinner />}
        <textarea value={text} onChange={e => setText(e.target.value)} />
      </div>
    )
  } else {
    result = (
      <div className="result-container">
        {isLoading && <Spinner />}
        <div className="result">
          {words.map((word, index) => (
            <WordChip
              key={index}
              word={word.text}
              logprob={word.logprob}
              threshold={threshold}
              replacements={word.replacements}
              onReplace={(newWord) => handleReplace(index, newWord)}
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <main>
      <h1>GPTed</h1>

      <details>
        <summary>Advanced settings</summary>
        <label>
          <strong>Threshold:</strong> <input type="number" step="1" value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
          <small>
            The <a href="https://en.wikipedia.org/wiki/Log_probability" target="_blank" rel="noreferrer">logprob</a> threshold.
            Tokens with logprobs smaller than this will be marked red.
          </small>
        </label>
        <label>
          <strong>Context:</strong> <small>Context for the text, which can help GPT3 better rank certain words.</small>
          <textarea placeholder="A short essay about picnics" value={context} onChange={e => setContext(e.target.value)} />
        </label>
        <label>
          <strong>Dictionary:</strong>
          <small>Known words or phrases. Helpful for uncommon or invented words and names.</small>
          <textarea placeholder="jujubu eschaton Frodo Baggins" value={wordlist} onChange={e => setWordlist(e.target.value)} />
        </label>
        <label>
          <strong>Show whole prompt:</strong> <input type="checkbox" checked={showWholePrompt} onChange={e => setShowWholePrompt(e.target.checked)} />
          <small>
            Show the whole prompt in the token view, instead of just your text. Mostly useful for debugging or curiosity.
          </small>
        </label>
      </details>

      <section id="inner">
        {result}
        <button onClick={toggleMode}>
          {mode === "edit" ? "Check" : "Edit"}
        </button>

        <p>
          <small>
            Based on <a href="https://github.com/vgel/gpted">GPTed</a> by <a href="https://vgel.me">Theia Vogel</a>.
            Made with React, Transformers, LLama 3.2, and transitively, most of the web.
            <br />
            This software is provided with absolutely no warranty.
          </small>
        </p>
      </section>
    </main>
  )
}
