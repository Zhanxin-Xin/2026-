"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const outDir = path.resolve(__dirname, "equations");
fs.mkdirSync(outDir, { recursive: true });

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svgOut = new SVG({ fontCache: "local" });
const doc = mathjax.document("", { InputJax: tex, OutputJax: svgOut });

function toSvg(latex) {
  const html = adaptor.outerHTML(doc.convert(latex, { display: true }));
  const a = html.indexOf("<svg");
  const b = html.indexOf("</svg>");
  let svg = html.slice(a, b + 6).replace(/<\?xml[^>]*>/g, "");
  if (!/xmlns="http:\/\/www\.w3\.org\/2000\/svg"/.test(svg)) {
    svg = svg.replace(/<svg /, '<svg xmlns="http://www.w3.org/2000/svg" ');
  }
  svg = svg.replace(/(width|height)="([0-9.]+)(ex|em)"/g, (_m, attr, num) => {
    const px = Math.round(parseFloat(num) * 9.2);
    return `${attr}="${px}px"`;
  });
  return svg.replace(/currentColor/g, "#000000");
}

const equations = {
  eq1: String.raw`I_k=\left[k\Delta t,\,\min\!\left((k+1)\Delta t,T\right)\right),\quad N=\min\!\left(70,\left\lceil T/\Delta t\right\rceil\right),\quad \Delta t=0.5\,\mathrm{s}.`,
  eq2: String.raw`\mathbf e_j=\frac{1}{|\mathcal S_j|}\sum_{r\in\mathcal S_j}\mathbf h_r,\qquad \mathbf e_j\in\mathbb R^{768}.`,
  eq3: String.raw`\mathbf x^{(T)}_k=\frac{\sum_j \ell_{jk}\mathbf e_j}{\sum_j \ell_{jk}},\qquad \ell_{jk}=\left|[a_j,b_j)\cap I_k\right|.`,
  eq4: String.raw`\mathbf x^{(A)}_k=\left[\boldsymbol\mu_k;\boldsymbol\sigma_k\right]\in\mathbb R^{50},\quad \boldsymbol\mu_k=\operatorname{mean}_{q\in I_k}\mathbf z_q,\quad \boldsymbol\sigma_k=\operatorname{std}_{q\in I_k}\mathbf z_q.`,
  eq5: String.raw`m^{(V)}_k=\mathbb I\!\left(s_k=1\right)\,\mathbb I\!\left(c_k\ge 0.8\right)\,\mathbb I\!\left(\mathbf x^{(V)}_k\ \text{is finite}\right).`,
  eq6: String.raw`\mathbf X^{(T)}\in\mathbb R^{70\times768},\qquad \mathbf X^{(A)}\in\mathbb R^{70\times50},\qquad \mathbf X^{(V)}\in\mathbb R^{70\times465}.`,
  eq7: String.raw`\widetilde{\mathbf x}^{(m)}_k=m^{(m)}_k\,m^{(P)}_k\,\mathbf x^{(m)}_k,\qquad m\in\{T,A,V\}.`
};

async function main() {
  for (const [name, latex] of Object.entries(equations)) {
    const svg = Buffer.from(toSvg(latex));
    await sharp(svg, { density: 320 }).png().toFile(path.join(outDir, `${name}.png`));
  }
}

main().catch((err) => {
  process.stderr.write(String(err));
  process.exit(1);
});
