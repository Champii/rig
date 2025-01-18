use anyhow::Result;
use rig::{
    completion::{ChatWithHistory, Message, ToolDefinition},
    providers,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = providers::openai::Client::from_env();

    // Create agent with history and a tool
    let mut agent = openai_client
        .agent(providers::openai::GPT_4O)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .tool(Adder)
        .build()
        .with_history();

    // First interaction
    let (response1, history1) = agent.chat_with_history("Calculate 2 + 5").await?;
    println!("First response: {}", response1);
    println!("\nHistory after first interaction:");
    for msg in history1 {
        println!("{}: {}", msg.role(), msg.content());
    }

    // Second interaction (asking about previous calculation)
    let (response2, history2) = agent
        .chat_with_history("What was the previous calculation?")
        .await?;
    println!("\nSecond response: {}", response2);
    println!("\nFull history:");
    for msg in history2 {
        let content = match &msg {
            Message::Chat { content, .. } => content,
            Message::ToolCall {
                name, arguments, ..
            } => &format!("Calling tool {} with arguments {}", name, arguments),
            Message::ToolResponse { content, .. } => content,
        };
        println!("{}: {}", msg.role(), content);
    }

    Ok(())
}
