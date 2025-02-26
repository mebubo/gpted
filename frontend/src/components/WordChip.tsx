import React, { useState, useEffect, useRef } from "react"
import { Replacement } from "../interfaces";

interface WordChipProps {
  word: string;
  logprob: number;
  threshold: number;
  replacements: Replacement[];
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

  console.log(`word: ->${word}<-`);

  let w1;
  let w2;
  let w3;
  // if word contains a newline, render a <br />
  if (word.includes("\n")) {
    [w1, w3] = word.split("\n");
    w2 = "\n";
    console.log(`split: ${w1} | ${w2} | ${w3}`);
  } else {
    w1 = word;
    w2 = "";
    w3 = "";
  }

  // sort replacements by logprob (make sure not to mutate the original array)
  const sortedReplacements = [...replacements].sort((a, b) => b.logprob - a.logprob)
  // convert logprobs to probabilities
  const withProbabilities = sortedReplacements.map(r => ({ ...r, probability: Math.exp(r.logprob)*100 }))

  return (
    <span
      title={logprob.toFixed(2)}
      className={`word-chip ${logprob < threshold ? "flagged" : ""}`}
      style={{ position: "relative", cursor: logprob < threshold ? "pointer" : "default" }}
      onClick={handleClick}
    >
      {w1}
      {w2 && <br />}
      {w3}
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
          {withProbabilities.map((option, index) => (
            <div
              key={index}
              onClick={() => handleReplacement(option.text)}
              onMouseEnter={() => setSelectedIndex(index)}
              style={{
                padding: "5px 10px",
                cursor: "pointer",
                color: "black",
                backgroundColor: selectedIndex === index ? "#f0f0f0" : "white",
                whiteSpace: "nowrap"
              }}
            >
              {option.text} <small style={{ fontSize: "0.7em", color: "#666" }}>{option.probability.toFixed(1)}%</small>
            </div>
          ))}
        </div>
      )}
    </span>
  )
}
