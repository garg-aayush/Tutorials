# Tools

1. [RepoMix](https://repomix.com/): Lets you take a github repo and concatenate all the files based on a pattern into a single file. This is useful for getting lots of context into a single file that your model can easily understand.
2. [GitMCP](https://gitmcp.io/): Use a github repo as a context source by creating an MCP server to let agents interact with the repo via tools. A good way to be able to let agents interact with the repo via tools. It also has a nice web chat interface.
3. [JINA AI](https://github.com/jina-ai/reader): turn any html page into a markdown file (much better than providing the html as context). To use, `https://r.jina.ai/https://your.url`
4. [Yazi](https://yazi-rs.github.io/): a fast terminal file manager with support for different file types and even images.
5. [Eza](https://eza.rocks/): a nice to have alternative to ls.
6. [Bat](https://github.com/sharkdp/bat): basically `cat` but with syntax highlighting and formatting.
7. [direnv](https://direnv.net/): loads and unloads environment variables automatically as you enter and leave a directory, driven by a per-folder `.envrc`.
8. [timg](https://github.com/hzeller/timg): shows images, animated gifs, videos, and PDFs directly in the terminal, using full-resolution graphics on iTerm2/kitty and half-block Unicode elsewhere. Handy for peeking at plots and screenshots without leaving the shell: `timg plot.png`, `timg --grid=3 *.jpg`, `timg -W --auto-crop doc.pdf`, or pipe into it with `echo "set terminal png; plot sin(x);" | gnuplot | timg -`. The `ils` alias in `setup-scripts/terminal/zshrc` is the README's "image ls" column view.
