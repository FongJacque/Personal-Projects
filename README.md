# Personal Projects

The source for my portfolio site, **[jacquefong.com](https://jacquefong.com)**, and
the projects it showcases. Each project's page shows its commit history and
runs its real code live in your browser.

## Projects

| Project | What it is | Try it |
|---|---|---|
| [Federated Lasso Regression](Python-Projects/Federated-Lasso-regression) | Three simulated data nodes train one shared sparse model without sharing raw data, compared against a centralized baseline. | [Run it](https://jacquefong.com/project.html?id=federated-lasso) |
| [Word Recognition Memory Experiment](Python-Projects/Word-Memory-Experiment) | A psychology experiment I designed and ran on whether people remember words better when they fit a scene, with the analysis of the results. | [Take a round](https://jacquefong.com/project.html?id=word-memory-experiment) |
| [Image Segmentation from Scratch](Computer-Vision-Projects/Image-Segmentation) | Felzenszwalb segmentation, SLIC superpixels and blob detection written without a computer vision library. | [Try your own photo](https://jacquefong.com/project.html?id=image-segmentation) |
| [Bingo Card Generator](Python-Projects/Bingo-Card-Generator) | Random bingo cards from numbers or your own word list. | [Make a card](https://jacquefong.com/project.html?id=bingo-card-generator) |

Each project folder has its own README with details and how to run it locally.

## How the site works

- **Plain HTML, CSS and JavaScript**, hosted on Cloudflare Pages. Every push to
  `main` publishes the site.
- **`assets/projects.json`** lists the projects. The Projects page sorts them by
  their latest commit, and each project page is built from that entry.
- **Live demos** run the project's own Python code in the visitor's browser with
  [Pyodide](https://pyodide.org) (R projects would use [webR](https://docs.r-wasm.org/webr/latest/)).
  Nothing is installed or uploaded, and nothing a visitor enters is stored.

## Repository layout

```
index.html, about.html,       site pages
projects.html, project.html,
resume.html, 404.html
assets/                       styles, images, résumé, demo runner, projects.json
docs/README.md                how to add a project to the site
Python-Projects/              one folder per project
Computer-Vision-Projects/
```

To add a project, see [docs/README.md](docs/README.md).

## Contact

[LinkedIn](https://www.linkedin.com/in/jacque-j-fong/)
