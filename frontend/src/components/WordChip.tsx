import React from "react"

export const WordChip = ({
  word,
  logprob,
  threshold,
  replacements,
  onReplace
}: {
  word: string,
  logprob: number,
  threshold: number,
  replacements: string[],
  onReplace: (newWord: string) => void
}) => {

  const handleReplacement = (event: React.ChangeEvent<HTMLSelectElement>) => {
    console.log("handleReplacement", event.target.value)
    const newWord = event.target.value
    onReplace(newWord)
  }

  const className = `word-chip ${logprob < threshold ? "flagged" : ""}`

  if (replacements.length === 0) {
    return <span className={className}>{word}</span>
  }

  return (
    <select
      className={className}
      onChange={handleReplacement}
      style={{
        appearance: 'none',
        border: 'none',
        fontFamily: 'inherit',
        fontSize: 'inherit',
        color: 'inherit',
        background: 'inherit',
        backgroundColor: 'red',
        padding: 0,
        cursor: 'pointer'
      }}
    >
      <option key="original" hidden>{word}</option>
      {replacements.map((r, i) => (
        <option key={i} value={r}>{r}</option>
      ))}
    </select>
  )
}
