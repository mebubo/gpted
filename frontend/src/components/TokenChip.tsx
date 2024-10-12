import React from "react"

export const TokenChip = ({ token, logprob, threshold }: { token: string, logprob: number, threshold: number }) => {
  return <span
    title={logprob.toFixed(2)}
    className={`token-chip ${logprob < threshold ? "flagged" : ""}`}
  >
    {token}
  </span>
}
