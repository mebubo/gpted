export interface Replacement {
  text: string
  logprob: number
}

export interface Word {
  text: string
  logprob: number
  replacements: Replacement[]
}
