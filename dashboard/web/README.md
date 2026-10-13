# Dashboard frontend

React, Vite, Tailwind, and shadcn/ui. The Python service serves the production build. Node is only
needed to rebuild.

```bash
npm install && npm run build
```

`npm run build` writes `dashboard/src/gpu_dashboard/static`. Geist and Geist Mono are the files in
`src/fonts` (SIL OFL, from [vercel/geist-font](https://github.com/vercel/geist-font)). See
`../README.md` for how to run the page and for the design notes.
