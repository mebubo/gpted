import * as esbuild from 'esbuild'

esbuild
  .build({
    entryPoints: [
      "./src/index.tsx",
    ],
    bundle: true,
    minify: false,
    sourcemap: process.env.NODE_ENV !== "production",
    target: ["chrome120", "firefox110"],
    outdir: "public",
    outbase: "src",
    define: {
      "process.env.NODE_ENV": `"${process.env.NODE_ENV}"`
    }
  })
  .catch(() => process.exit(1))
