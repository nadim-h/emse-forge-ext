You are an expert software testing engineer. You read a program carefully and design test input/output that covers its behavior.

---

Below is a complete $langname program that reads from standard input and writes to standard output.

```$lang
$code
```

Write up to $n test cases for this program.
Aim for good coverage of the programs behaviour.

Requirements:
- `input` must be the exact, complete text fed to stdin.
- `output` must be exactly what THIS program prints for that input. Trace the code to work it out. Do not guess, and do not describe what the program "should" do.
- Respect the input format the code parses; malformed input makes the test useless.

Provide a fenced JSON array with the input/output pairs at the end of your response:

```json
[
  {"input": "...", "output": "..."},
  ...
]
```

Use `\n` inside the JSON strings for newlines.
