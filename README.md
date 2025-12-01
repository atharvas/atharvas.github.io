~This is a single HTML and CSS file that serves as my personal website. No NPM, no JavaScript, no bullScript.~

This is an Astro template for an easy-to-maintain personal website.


## Installation:
Clone the repo and run:

```bash
$ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
$ nvm install --lts
$ nvm use --lts
$ npm install
$ npm run dev --verbose
```

## Customization:

Edit the following files to customize your site:
 - `src/pages/index.astro` - Main content of your website
 - `src/data/BioLinks.ts` - Links to co-authors' profiles.
 - `src/data/News.ts` - New items to display on your site.
 - `src/data/Papers.ts` - List of publications.


## Deployment:

Once you're done, simply push to your GitHub repository. A github action should deploy the site to GitHub pages automatically.