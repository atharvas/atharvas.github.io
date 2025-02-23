/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{astro,html,js,jsx,tsx,md,mdx}', // <--- add whatever file extensions you use
    './public/**/*.html',
  ],
  theme: {
    extend: {},
  },
  plugins: [
    "@tailwindcss/typography",
  ],
};
