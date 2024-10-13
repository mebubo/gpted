import React, { useState, useEffect, useRef } from "react"

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
          ref={dropdownRef}
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
