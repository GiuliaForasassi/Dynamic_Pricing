# Copilot Chat Formatting

- When writing mathematical notation in chat responses or Markdown, use `$...$` for inline math and `$$...$$` for display math. Put display delimiters on their own lines.
- Do not emit raw LaTeX or use `\[...\]` / `\(...\)` delimiters; VS Code Copilot Chat may display those literally.
- If the notation is not needed, explain it in prose. Use a fenced code block only when the user needs to see literal LaTeX source.

For example, write:

$$
p_{\text{effective}} =
\begin{cases}
p_{\text{model}}, & \text{if eligible} \\
0, & \text{otherwise}
\end{cases}
$$