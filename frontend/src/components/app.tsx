import React, { useState } from "react"
import { TokenChip } from "./TokenChip"

interface Word {
  text: string
  logprob: number
}

async function checkText(text: string): Promise<Word[]> {
  await new Promise(resolve => setTimeout(resolve, 3000));

  const words = text.split(/\b/)
  return words.map(word => ({ text: word, logprob: -word.length }))
}

export default function App() {
  const [threshold, setThreshold] = useState(-5.0)
  const [context, setContext] = useState("")
  const [wordlist, setWordlist] = useState("")
  const [showWholePrompt, setShowWholePrompt] = useState(false)
  const [text, setText] = useState("")
  const [mode, setMode] = useState<"edit" | "check">("edit")
  const [words, setWords] = useState<Word[]>([])

  const toggleMode = async () => {
    if (mode === "edit") {
      const checkedWords = await checkText(text)
      setWords(checkedWords)
      setMode("check")
    } else {
      setMode("edit")
    }
  }

  let result

  if (mode === "edit") {
    result = <textarea value={text} onChange={e => setText(e.target.value)} />
  } else {
    result = (
      <div className="result">
        {words.map((word, index) => (
          <TokenChip
            key={index}
            token={word.text}
            logprob={word.logprob}
            threshold={threshold}
          />
        ))}
      </div>
    )
  }

  return (
    <main>
      <h1>GPTed</h1>

      <details>
        <summary>Advanced settings</summary>
        <label>
          <strong>Threshold:</strong> <input type="number" step="0.1" value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
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
            Made by <a href="https://vgel.me">Theia Vogel</a> (<a href="https://twitter.com/voooooogel">@voooooogel</a>).
            Made with Svelte, GPT-3, and transitively, most of the web.
            <br />
            This software is provided with absolutely no warranty.
          </small>
        </p>
      </section>
    </main>
  )
}
