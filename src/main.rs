use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentState {
    Plan,
    Act,
    Observe,
    Evaluate,
    Done,
}

#[derive(Debug, Default)]
struct Memory {
    plan: Option<String>,
    completion_criteria: Vec<String>,
    files_created: Vec<String>,
    files_modified: Vec<String>,
    last_errors: Vec<String>,
    test_results: Vec<String>,
    last_tool_output: Option<String>,
}

#[derive(Debug, Clone)]
struct ExecResult {
    status_success: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = "sk-or-REPLACE_ME".to_string();

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    run_agent(&client, args.prompt).await?;

    Ok(())
}

async fn run_agent(
    client: &Client<OpenAIConfig>,
    prompt: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut messages = vec![
        json!({
          "role": "system",
          "content": "You are a structured autonomous coding agent.\n\
        Follow this strict state machine: PLAN -> ACT -> OBSERVE -> EVALUATE -> DONE.\n\
        Rules:\n\
        - First iteration MUST produce a concrete step-by-step plan and completion criteria.\n\
        - You may call tools only in ACT.\n\
        - You may reason only in PLAN/EVALUATE.\n\
        - Before each Bash command include self-critique: risks, dependency check, safety check.\n\
        - Use exit codes and explicit assertions from execution output; avoid fragile substring checks.\n\
        - Use tools only when changing or inspecting environment; pure reasoning is allowed without tools.\n\
        - Prefer Patch over full-file Write when editing existing files.\n\
        - Return DONE only when all completion criteria are satisfied and validated."
        }),
        json!({ "role": "user", "content": prompt }),
    ];

    let mut state = AgentState::Plan;
    let mut iterations = 0;
    let max_iterations = 20;
    let mut memory = Memory::default();

    while iterations < max_iterations && state != AgentState::Done {
        iterations += 1;
        println!("\n=== ITERATION {} | STATE {:?} ===", iterations, state);

        let state_prompt = make_state_prompt(state, &memory);
        messages.push(json!({ "role": "user", "content": state_prompt }));

        let response: Value = match client
            .chat()
            .create_byot(json!({
              "model": "openai/gpt-oss-120b:free",
              "messages": messages,
              "tools": [
                tool_schema("Read", &["file_path"]),
                tool_schema("Write", &["file_path", "content"]),
                tool_schema("Patch", &["file_path", "diff"]),
                tool_schema("ListFiles", &["path"]),
                tool_schema("Tree", &["path"]),
                tool_schema("Bash", &["command"])
              ]
            }))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("LLM error: {e}");
                break;
            }
        };

        let message = &response["choices"][0]["message"];
        messages.push(message.clone());

        if let Some(content) = message["content"].as_str() {
            if state == AgentState::Plan {
                memory.plan = Some(content.to_string());
                memory.completion_criteria = extract_bullets(content);
            }
            if state == AgentState::Evaluate && content.contains("DONE") {
                state = AgentState::Done;
                println!("✅ Agent reported DONE");
                continue;
            }
        }

        let tool_calls = message["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        match state {
            AgentState::Plan => {
                if !tool_calls.is_empty() {
                    messages.push(json!({"role":"user","content":"Invalid transition: PLAN cannot call tools. Provide plan and criteria only."}));
                    continue;
                }
                state = AgentState::Act;
            }
            AgentState::Act => {
                if tool_calls.is_empty() {
                    messages.push(
                        json!({"role":"user","content":"ACT requires at least one tool call."}),
                    );
                    continue;
                }

                for tool_call in tool_calls {
                    let id = tool_call["id"].as_str().unwrap_or_default();
                    let name = tool_call["function"]["name"].as_str().unwrap_or_default();
                    let args_str = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    let args: Value = match serde_json::from_str(args_str) {
                        Ok(v) => v,
                        Err(e) => {
                            messages.push(
                                json!({"role":"user","content":format!("Invalid JSON args: {e}")}),
                            );
                            continue;
                        }
                    };

                    let result = execute_tool(name, &args, &mut memory)?;
                    memory.last_tool_output = Some(result.clone());

                    messages.push(json!({"role":"tool","tool_call_id":id,"content":result}));
                }
                state = AgentState::Observe;
            }
            AgentState::Observe => {
                if !tool_calls.is_empty() {
                    messages.push(json!({"role":"user","content":"Invalid transition: OBSERVE cannot call tools. Summarize observations only."}));
                    continue;
                }
                state = AgentState::Evaluate;
            }
            AgentState::Evaluate => {
                if !tool_calls.is_empty() {
                    messages.push(json!({"role":"user","content":"Invalid transition: EVALUATE cannot call tools. Decide ACT vs DONE with explicit criteria check."}));
                    continue;
                }
                state = AgentState::Act;
            }
            AgentState::Done => {}
        }
    }

    if iterations >= max_iterations {
        eprintln!("❌ Max iterations reached");
    }

    Ok(())
}

fn make_state_prompt(state: AgentState, memory: &Memory) -> String {
    let mem = format!(
        "Memory summary:\nPlan: {:?}\nCompletion: {:?}\nFiles created: {:?}\nFiles modified: {:?}\nErrors: {:?}\nTests: {:?}",
        memory.plan,
        memory.completion_criteria,
        memory.files_created,
        memory.files_modified,
        memory.last_errors,
        memory.test_results
    );

    match state {
        AgentState::Plan => format!(
            "STATE=PLAN\nCreate numbered implementation plan and explicit completion criteria.\n{mem}"
        ),
        AgentState::Act => format!(
            "STATE=ACT\nExecute next steps with tools. Before Bash, include self-critique (risk, deps, safety).\n{mem}"
        ),
        AgentState::Observe => {
            format!("STATE=OBSERVE\nSummarize tool outputs and facts only.\n{mem}")
        }
        AgentState::Evaluate => format!(
            "STATE=EVALUATE\nCheck completion criteria explicitly with pass/fail per criterion. Reply DONE if all pass; otherwise explain what remains.\n{mem}"
        ),
        AgentState::Done => "STATE=DONE".to_string(),
    }
}

fn extract_bullets(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with('-') || t.starts_with('*') || t.starts_with("1.") {
                Some(t.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn tool_schema(name: &str, params: &[&str]) -> Value {
    let mut props = serde_json::Map::new();
    let mut required = vec![];

    for p in params {
        props.insert(p.to_string(), json!({ "type": "string" }));
        required.push(p.to_string());
    }

    json!({
      "type": "function",
      "function": {
        "name": name,
        "description": format!("Tool: {}", name),
        "parameters": {
          "type": "object",
          "properties": props,
          "required": required
        }
      }
    })
}

fn execute_tool(
    name: &str,
    args: &Value,
    memory: &mut Memory,
) -> Result<String, Box<dyn std::error::Error>> {
    match name {
        "Read" => {
            let file_path = str_arg(args, "file_path")?;
            Ok(std::fs::read_to_string(file_path)?)
        }
        "Write" => {
            let file_path = str_arg(args, "file_path")?;
            let content = str_arg(args, "content")?;
            let existed = Path::new(file_path).exists();
            std::fs::write(file_path, content)?;
            if existed {
                push_unique(&mut memory.files_modified, file_path);
            } else {
                push_unique(&mut memory.files_created, file_path);
            }
            Ok("file written".to_string())
        }
        "Patch" => {
            let file_path = str_arg(args, "file_path")?;
            let diff = str_arg(args, "diff")?;
            apply_simple_patch(file_path, diff)?;
            push_unique(&mut memory.files_modified, file_path);
            Ok("patch applied".to_string())
        }
        "ListFiles" => {
            let path = str_arg(args, "path")?;
            let mut files = vec![];
            for entry in std::fs::read_dir(path)? {
                let e = entry?;
                files.push(e.file_name().to_string_lossy().to_string());
            }
            files.sort();
            Ok(files.join("\n"))
        }
        "Tree" => {
            let path = str_arg(args, "path")?;
            let mut out = vec![];
            tree_walk(Path::new(path), 0, &mut out)?;
            Ok(out.join("\n"))
        }
        "Bash" => {
            let command = str_arg(args, "command")?;
            enforce_safe_command(command)?;
            let exec = run_bash(command)?;
            let summary = format!(
                "Execution status: {}\nExit code: {:?}\nSTDOUT:\n{}\nSTDERR:\n{}",
                if exec.status_success {
                    "SUCCESS"
                } else {
                    "FAILURE"
                },
                exec.exit_code,
                exec.stdout,
                exec.stderr
            );
            memory.test_results.push(summary.clone());
            if !exec.status_success {
                let hint = classify_failure(&exec.stderr, &exec.stdout);
                memory.last_errors.push(hint.to_string());
            }
            Ok(summary)
        }
        _ => Ok(format!("unknown tool: {}", name)),
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing {key}").into())
}

fn run_bash(command: &str) -> Result<ExecResult, Box<dyn std::error::Error>> {
    let output: Output = Command::new("sh").arg("-c").arg(command).output()?;
    Ok(ExecResult {
        status_success: output.status.success(),
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn classify_failure(stderr: &str, stdout: &str) -> &'static str {
    let all = format!("{}\n{}", stderr.to_lowercase(), stdout.to_lowercase());
    if all.contains("not found") || all.contains("no such file") {
        "Missing dependency or command"
    } else if all.contains("syntax") {
        "Syntax error"
    } else {
        "Logic/runtime failure"
    }
}

fn enforce_safe_command(command: &str) -> Result<(), Box<dyn std::error::Error>> {
    let blocked = ["rm -rf /", "mkfs", "shutdown", "reboot", ":(){:|:&};:"];
    if blocked.iter().any(|b| command.contains(b)) {
        return Err("blocked unsafe command".into());
    }
    Ok(())
}

fn apply_simple_patch(file_path: &str, diff: &str) -> Result<(), Box<dyn std::error::Error>> {
    let original = std::fs::read_to_string(file_path)?;
    let mut updated = original.clone();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("- ") {
            updated = updated.replace(rest, "");
        } else if let Some(rest) = line.strip_prefix("+ ") {
            updated.push_str(rest);
            updated.push('\n');
        }
    }
    std::fs::write(file_path, updated)?;
    Ok(())
}

fn tree_walk(
    path: &Path,
    depth: usize,
    out: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let name = path
        .file_name()
        .map(|v| v.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    out.push(format!("{}{}", "  ".repeat(depth), name));

    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|d| d.path()))
            .collect();
        entries.sort();
        for entry in entries {
            tree_walk(&entry, depth + 1, out)?;
        }
    }

    Ok(())
}

fn push_unique(items: &mut Vec<String>, value: &str) {
    if !items.iter().any(|v| v == value) {
        items.push(value.to_string());
    }
}
