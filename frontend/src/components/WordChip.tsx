import React, { useState, useEffect, useRef } from "react"

interface WordChipProps {
  word: string;
  logprob: number;
  threshold: number;
  replacements: string[];
  onReplace: (newWord: string) => Promise<void>;
}

export function WordChip({
  word,
  logprob,
  threshold,
  replacements,
  onReplace
}: WordChipProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  const handleReplacement = async (newWord: string) => {
    console.log("handleReplacement", newWord);
    await onReplace(newWord);
    setIsExpanded(false);  // Close the dropdown
  };

  return (
    <span
      title={logprob.toFixed(2)}
      className={`word-chip ${logprob < threshold ? "flagged" : ""}`}
      style={{ position: "relative", cursor: logprob < threshold ? "pointer" : "default" }}
      onClick={handleClick}
    >
      {word}
      {isExpanded && (
        <div
          ref={dropdownRef}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            zIndex: 100,
            maxHeight: "400px",
            overflowY: "auto",
            backgroundColor: "white",
            border: "1px solid #ccc",
            borderRadius: "4px",
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)"
          }}
        >
          {replacements.map((option, index) => (
            <div
              key={index}
              onClick={() => handleReplacement(option)}
              onMouseEnter={() => setSelectedIndex(index)}
              style={{
                padding: "5px 10px",
                cursor: "pointer",
                color: "black",
                backgroundColor: selectedIndex === index ? "#f0f0f0" : "white"
              }}
            >
              {option}
            </div>
          ))}
        </div>
      )}
    </span>
  )
}
