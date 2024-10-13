import React, { useState } from "react"

import React, { useState } from "react"

export const TokenChip = ({
  token,
  logprob,
  threshold,
  replacements,
  onReplace
}: {
  token: string,
  logprob: number,
  threshold: number,
  replacements: string[],
  onReplace: (newToken: string) => void
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleClick = () => {
    if (logprob < threshold && replacements.length > 0) {
      setIsExpanded(true);
    }
  }

  const handleReplacement = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newToken = event.target.value
    if (newToken !== token) {
      onReplace(newToken)
    }
    setIsExpanded(false);
  }

  return (
    <span
      title={logprob.toFixed(2)}
      className={`token-chip ${logprob < threshold ? "flagged" : ""}`}
      style={{ position: "relative", cursor: logprob < threshold ? "pointer" : "default" }}
      onClick={handleClick}
    >
      {token}
      {isExpanded && (
        <select
          onChange={handleReplacement}
          value={token}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            zIndex: 1000
          }}
          size={replacements.length}
          autoFocus
        >
          {replacements.map((rep, idx) => (
            <option key={idx} value={rep}>{rep}</option>
          ))}
        </select>
      )}
    </span>
  )
}
