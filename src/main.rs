use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::env;
use std::process::Command;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
  #[arg(short = 'p', long)]
  prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  let base_url = env::var("OPENROUTER_BASE_URL")
  .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

  let api_key = "sk-or-REPLACE_ME".to_string(); // ⚠️ don't hardcode real keys

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
      "content": "You are a strict autonomous coding agent.\n\
- You MUST use tools.\n\
- NEVER answer without executing tools.\n\
- You MUST verify correctness using execution output.\n\
- If output is incorrect, fix and retry.\n\
- Only finish when task is fully correct.\n\
- Final answer MUST be EXACTLY: SUCCESS"
    }),
    json!({ "role": "user", "content": prompt })
  ];

  let mut iterations = 0;
  let max_iterations = 15;

  let mut tool_used = false;
  let mut saw_bash = false;

  loop {
    if iterations >= max_iterations {
      eprintln!("❌ Max iterations reached");
      break;
    }

    iterations += 1;

    println!("\n==============================");
    println!("🔁 ITERATION {}", iterations);
    println!("==============================");

    let response_result = client
    .chat()
    .create_byot(json!({
      "model": "openai/gpt-oss-120b:free",
      "messages": messages,
      "tools": [
        tool_schema("Read", &["file_path"]),
                       tool_schema("Write", &["file_path", "content"]),
                       tool_schema("Bash", &["command"])
      ]
    }))
    .await;

    let response: Value = match response_result {
      Ok(r) => r,
      Err(e) => {
        eprintln!("❌ LLM error: {}", e);
        break;
      }
    };

    let message = &response["choices"][0]["message"];

    messages.push(message.clone());

    // Print assistant text (if any)
    if let Some(content) = message["content"].as_str() {
      println!("🧠 Assistant: {}", content);

      if content.trim() == "SUCCESS" {
        println!("\n✅ TASK COMPLETED SUCCESSFULLY");
        break;
      }
    }

    let tool_calls = match message["tool_calls"].as_array() {
      Some(t) => t,
      None => {
        // force tool usage
        if !tool_used {
          messages.push(json!({
            "role": "user",
            "content": "You must use tools. Continue."
          }));
          continue;
        }

        // force execution
        if !saw_bash {
          messages.push(json!({
            "role": "user",
            "content": "You must run the program using Bash."
          }));
          continue;
        }

        messages.push(json!({
          "role": "user",
          "content": "Not accepted. Continue fixing using tools."
        }));

        continue;
      }
    };

    for tool_call in tool_calls {
      let id = match tool_call["id"].as_str() {
        Some(v) => v,
        None => continue,
      };

      let name = match tool_call["function"]["name"].as_str() {
        Some(v) => v,
        None => continue,
      };

      let args_str = match tool_call["function"]["arguments"].as_str() {
        Some(v) => v,
        None => continue,
      };

      println!("\n🔧 TOOL CALL → {}", name);
      println!("ARGS: {}", args_str);

      let args: Value = match serde_json::from_str(args_str) {
        Ok(v) => v,
        Err(e) => {
          eprintln!("❌ Bad JSON args: {}", e);

          messages.push(json!({
            "role": "user",
            "content": "Tool arguments invalid JSON. Fix them."
          }));

          continue;
        }
      };

      let result = match execute_tool(name, &args) {
        Ok(r) => r,
        Err(e) => {
          eprintln!("❌ Tool error: {}", e);

          messages.push(json!({
            "role": "user",
            "content": format!("Tool {} failed: {}. Fix it.", name, e)
          }));

          continue;
        }
      };

      println!("📤 TOOL RESULT ({}) ↓\n{}\n", name, result);

      tool_used = true;

      if name == "Bash" {
        saw_bash = true;
      }

      messages.push(json!({
        "role": "tool",
        "tool_call_id": id,
        "content": result
      }));

      // 🔥 after writing → force run
      if name == "Write" {
        messages.push(json!({
          "role": "user",
          "content": "Run the program using Bash."
        }));
      }

      // 🔥 smart evaluation
      if name == "Bash" {
        let has_error =
        result.contains("Traceback") ||
        result.contains("error") ||
        result.contains("Exception") ||
        result.contains("failed");

        if has_error {
          messages.push(json!({
            "role": "user",
            "content": format!(
              "The program failed.\n\nOutput:\n{}\n\nFix and retry.",
              result
            )
          }));
        } else {
          messages.push(json!({
            "role": "user",
            "content": format!(
              "Program output:\n{}\n\n\
Evaluate strictly:\n\
- Does it satisfy the task fully?\n\
- Correct behavior?\n\
- Any logical issues?\n\n\
If NOT correct:\n\
- Fix code using Write\n\
- Run again using Bash\n\n\
If correct:\n\
Respond EXACTLY with: SUCCESS",
result
            )
          }));
        }
      }
    }
  }

  Ok(())
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

fn execute_tool(name: &str, args: &Value) -> Result<String, Box<dyn std::error::Error>> {
  match name {
    "Read" => {
      let file_path = args.get("file_path")
      .and_then(|v| v.as_str())
      .ok_or("missing file_path")?;

      Ok(std::fs::read_to_string(file_path)?)
    }

    "Write" => {
      let file_path = args.get("file_path")
      .and_then(|v| v.as_str())
      .ok_or("missing file_path")?;

      let content = args.get("content")
      .and_then(|v| v.as_str())
      .ok_or("missing content")?;

      std::fs::write(file_path, content)?;
      Ok("file written".to_string())
    }

    "Bash" => {
      let command = args.get("command")
      .and_then(|v| v.as_str())
      .ok_or("missing command")?;

      let output = Command::new("sh")
      .arg("-c")
      .arg(command)
      .output()?;

      let stdout = String::from_utf8_lossy(&output.stdout);
      let stderr = String::from_utf8_lossy(&output.stderr);

      Ok(format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr))
    }

    _ => Ok(format!("unknown tool: {}", name)),
  }
}
