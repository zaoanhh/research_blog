<img align="right" width="150" alt="logo" src="https://user-images.githubusercontent.com/5889006/190859553-5b229b4f-c476-4cbd-928f-890f5265ca4c.png">

# Hugo Theme Stack Starter Template

This is a quick start template for [Hugo theme Stack](https://github.com/CaiJimmy/hugo-theme-stack). It uses [Hugo modules](https://gohugo.io/hugo-modules/) feature to load the theme.

It comes with a basic theme structure and configuration. **This repository builds and deploys locally**: the site is rendered on your machine and pushed to the `gh-pages` branch, which GitHub Pages serves. GitHub Actions is no longer needed; the workflow files are kept as manual fallbacks only.

## Get started

1. Click *Use this template*, and create your repository as `<username>.github.io` on GitHub.
![Step 1](https://user-images.githubusercontent.com/5889006/156916624-20b2a784-f3a9-4718-aa5f-ce2a436b241f.png)

2. Once the repository is created, create a GitHub codespace associated with it.
![Create codespace](https://user-images.githubusercontent.com/5889006/156916672-43b7b6e9-4ffb-4704-b4ba-d5ca40ffcae7.png)

3. And voila! You're ready to go. The codespace has been configured with the latest version of Hugo extended, just run `hugo server` in the terminal and see your new site in action.

4. Check `config` folder for the configuration files. You can edit them to suit your needs. Make sure to update the `baseURL` property in `config/_default/config.toml` to your site's URL.

5. Open Settings -> Pages. Change the build branch from `master` to `gh-pages`.
![Build](https://github.com/namanh11611/hugo-theme-stack-starter/assets/16586200/12c763cd-bead-4923-b610-8788f388fcb5)

6. Once you're done editing the site, commit it, push `master`, then publish the rendered site from your machine:

```bash
./scripts/deploy.sh
```

The script builds with `hugo --minify --gc`, stages the output in a temporary git worktree, and force-pushes it to `gh-pages` as a single fresh commit, then waits for the GitHub Pages build to finish. See [Local build & deploy](#local-build--deploy) for details.

---

In case you don't want to use GitHub codespace, you can also run this template in your local machine. **You need to install Git, Go and Hugo extended (v0.157.0 or newer) locally.**

## Local build & deploy

Everything happens locally; no CI runner or Actions minutes are used.

Requirements: Git, Go, rsync, and Hugo **extended** >= 0.157, plus push credentials for `origin` (`gh auth login` or a git credential helper).

```bash
./scripts/deploy.sh                 # build + deploy to gh-pages + wait for Pages
./scripts/deploy.sh --skip-build    # publish the existing public/ as-is
./scripts/deploy.sh --no-wait       # do not poll the Pages build status
./scripts/deploy.sh -m "message"    # custom commit message
```

What it does:

1. `hugo --minify --gc --cleanDestinationDir` renders the site into `public/`.
2. The output is synced into `.deploy-gh-pages/`, a temporary git worktree (ignored by git).
3. A parentless commit is created and force-pushed to `gh-pages`, so the branch always contains exactly one commit with the generated files. If the output is identical to what is already published, the push is skipped.
4. When `gh` is available, the script polls the Pages build API and prints the live URL.

Preview locally with `hugo server -D`; build only with `hugo --minify --gc --cleanDestinationDir`.

GitHub Pages must be configured once under *Settings -> Pages*: source **Deploy from a branch**, branch `gh-pages`, folder `/ (root)`. This repository is already configured that way.

## Update theme

```bash
./scripts/update-theme.sh
```

It runs `hugo mod get -u` + `hugo mod tidy`, verifies the site still builds, and leaves the `go.mod`/`go.sum` changes for you to review and commit. (The old daily workflow is kept as a manual fallback only.)

> This starter template has been configured with `v4` version of theme. Due to the limitation of Go module, once the `v5` or up version of theme is released, you need to update the theme manually. (Modifying `config/module.toml` file)

## Deploy to another static page hostings

If you want to build this site using another static page hosting, you need to make sure they have Go installed in the machine. 

<details>
  <summary>Vercel</summary>
  
You need to overwrite build command to install manually Go:

```
amazon-linux-extras install golang1.11 && hugo --gc --minify
```

![](https://user-images.githubusercontent.com/5889006/156917172-01e4d418-3469-4ffb-97e4-a905d28b8424.png)

If you are using Node.js 20, you need to overwrite the install command to install manually Go:

```
dnf install -y golang
```

![image](https://github.com/zhi-yi-huang/hugo-theme-stack-starter/assets/83860323/777c1109-dfc8-4893-9db7-1305ec027cf5)


Make sure also to specify Hugo version in the environment variable `HUGO_VERSION` (Use the latest version of Hugo extended):

![Environment variable](https://user-images.githubusercontent.com/5889006/156917212-afb7c70d-ab85-480f-8288-b15781a462c0.png)
</details>
