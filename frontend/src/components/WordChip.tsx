import React, { useState, useEffect, useRef } from "react"

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
  const [isExpanded, setIsExpanded] = useState(false);
  const dropdownRef = useRef<HTMLSelectElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsExpanded(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleClick = () => {
    if (logprob < threshold && replacements.length > 0) {
      setIsExpanded(true);
    }
  }

  const handleReplacement = (event: React.ChangeEvent<HTMLSelectElement>) => {
    console.log("handleReplacement", event.target.value)
    const newWord = event.target.value
    onReplace(newWord)
    setIsExpanded(false);
  }

  return (
    <span
      title={logprob.toFixed(2)}
      className={`word-chip ${logprob < threshold ? "flagged" : ""}`}
      style={{ position: "relative", cursor: logprob < threshold ? "pointer" : "default" }}
      onClick={handleClick}
    >
      {word}
      {isExpanded && (
        <select
          ref={dropdownRef}
          onChange={handleReplacement}
          value={word}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            zIndex: 1000,
            overflowY: "auto"
          }}
          size={replacements.length + 1}
        >
          <option key="original" hidden>{word}</option>
          {replacements.map((r, i) => (
            <option key={i} value={r}>{r}</option>
          ))}
        </select>
      )}
    </span>
  )
}
