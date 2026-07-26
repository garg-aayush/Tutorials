# Copying Cursor Settings from One Machine to Another

## Source Machine (Linux/Mac)

Export the list of installed extensions:
```bash
cursor --list-extensions > extensions.txt
```

Copy these files from the Cursor config directory:
- **Linux:** `~/.config/Cursor/User/`
- **Mac:** `~/Library/Application Support/Cursor/User/`

```bash
cp ~/.config/Cursor/User/settings.json .       # or Mac path
cp ~/.config/Cursor/User/keybindings.json .
```

## Target: Linux/Mac

Copy settings and keybindings to the config directory:
```bash
cp settings.json ~/.config/Cursor/User/settings.json
cp keybindings.json ~/.config/Cursor/User/keybindings.json
```

Install all extensions:
```bash
cat extensions.txt | xargs -L 1 cursor --install-extension
```

## Target: Windows (Git Bash)

Copy settings and keybindings:
```bash
cp settings.json "$APPDATA/Cursor/User/settings.json"
cp keybindings.json "$APPDATA/Cursor/User/keybindings.json"
```

Install all extensions:
```bash
cat extensions.txt | xargs -L 1 cursor --install-extension
```

If `cursor` is not in PATH, use the full path:
```bash
cat extensions.txt | xargs -L 1 "/c/Users/$USERNAME/AppData/Local/Programs/Cursor/resources/app/bin/cursor.cmd" --install-extension
```

Restart Cursor after all steps.
