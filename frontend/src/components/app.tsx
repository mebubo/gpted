import React, { useState } from "react"

export default function App() {
  const [threshold, setThreshold] = useState(0.0)
  const [context, setContext] = useState("")
  const [wordlist, setWordlist] = useState("")
  const [showWholePrompt, setShowWholePrompt] = useState(false)
  const [checked, setChecked] = useState(false)
  const [text, setText] = useState("")

  const click = () => {
    setChecked(true)
  }

  let result

  if (checked === null) {
    result = <textarea value={text} />
  } else {
    result = <div className="result">
      {text}
    </div>
  }

  return (
    <>
      <h1>GPTed</h1>

      <details>
        <summary>Advanced settings</summary>
        <label>
          <strong>Threshold:</strong> <input type="number" step="0.1" value={threshold} />
          <small>
            The <a href="https://en.wikipedia.org/wiki/Log_probability" target="_blank" rel="noreferrer">logprob</a> threshold.
            Tokens with logprobs smaller than this will be marked red.
          </small>
        </label>
        <label>
          <strong>Context:</strong> <small>Context for the text, which can help GPT3 better rank certain words.</small>
          <textarea placeholder="A short essay about picnics" value={context} />
        </label>
        <label>
          <strong>Dictionary:</strong>
          <small>Known words or phrases. Helpful for uncommon or invented words and names.</small>
          <textarea placeholder="jujubu eschaton Frodo Baggins" value={wordlist} />
        </label>
        <label>
          <strong>Show whole prompt:</strong> <input type="checkbox" checked={showWholePrompt} />
          <small>
            Show the whole prompt in the token view, instead of just your text. Mostly useful for debugging or curiosity.
          </small>
        </label>
      </details>

      <section id="inner">
        {result}
        <button onClick={click}>
          {checked ? "Check" : "Edit"}
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
    </>
  )
}
