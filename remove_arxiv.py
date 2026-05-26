import re

bib_file = "manuscript/references.bib"
tex_file = "manuscript/main.tex"

with open(bib_file, "r") as f:
    bib_content = f.read()

# Find all entries with "arXiv preprint"
# Regex to match bibtex entries: @type{key, ... }
entry_pattern = re.compile(r'@\w+\s*\{\s*([^,]+),.*?\n\}', re.DOTALL)
arxiv_keys = []

new_bib_content = bib_content

entries = list(re.finditer(r'@\w+\s*\{\s*([^,]+),.*?^\}', bib_content, flags=re.DOTALL | re.MULTILINE))
for match in entries:
    entry_text = match.group(0)
    key = match.group(1).strip()
    if 'arXiv preprint' in entry_text:
        arxiv_keys.append(key)
        new_bib_content = new_bib_content.replace(entry_text, "")

# remove extra newlines
new_bib_content = re.sub(r'\n{3,}', '\n\n', new_bib_content)

with open(bib_file, "w") as f:
    f.write(new_bib_content)

print(f"Removed {len(arxiv_keys)} arXiv keys: {arxiv_keys}")

with open(tex_file, "r") as f:
    tex_content = f.read()

def repl_cite(match):
    cite_cmd = match.group(1) # e.g. \cite or \citep
    keys_str = match.group(2)
    keys = [k.strip() for k in keys_str.split(",")]
    new_keys = [k for k in keys if k not in arxiv_keys]
    if not new_keys:
        return "" # remove the cite command completely
    # return the cite command with remaining keys
    return f"{cite_cmd}{{{','.join(new_keys)}}}"

# Regex for \cite{...} or \citep{...}
# Assuming cite commands are like \cite{key1,key2}
new_tex_content = re.sub(r'(\\cite[a-zA-Z]*)\{([^}]+)\}', repl_cite, tex_content)

# Clean up any empty cites or double spaces left behind (like " . " if cite was removed at the end of sentence)
new_tex_content = re.sub(r'~\s*\.', '.', new_tex_content)
new_tex_content = re.sub(r'\s+~', ' ', new_tex_content)
new_tex_content = new_tex_content.replace("~", "~") # just to be safe, \cite usually preceded by ~

with open(tex_file, "w") as f:
    f.write(new_tex_content)
print("Updated main.tex")
