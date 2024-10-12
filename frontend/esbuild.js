import * as esbuild from 'esbuild'

const ctx = await esbuild.context({
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

if (process.argv.includes('--watch')) {
  await ctx.watch()
  console.log('Watching...')
} else {
  await ctx.rebuild()
  await ctx.dispose()
}
