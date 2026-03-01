from .VARS import *
from .utils import extract_and_validate_answer
from openai import OpenAI
import logging
import time
import re


def send_request(
    problem, client, model, enable_thinking, thinking_interven, think_content=None
):
    messages = [
        {
            "role": "system",
            "content": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
        },
        {
            "role": "user",
            "content": problem,
        },
    ]

    chat_template_kwargs = {"top_k": 20, "skip_special_tokens": False}
    sample_params = {}
    extra_body = {}
    if enable_thinking:
        sample_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": MAX_NEW_TOKENS - MAX_NON_THINKING_TOKENS,
        }
        extra_body = chat_template_kwargs

    elif think_content:
        messages = think_content + messages
        sample_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": MAX_NON_THINKING_TOKENS,
        }
        extra_body = chat_template_kwargs | {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    if enable_thinking and thinking_interven:
        extra_body = chat_template_kwargs | {
            "add_generation_prompt": False,
            "continue_final_message": True,
        }
    # Define retry parameters
    max_retries = 5
    retry_delay_seconds = 120
    attempt = 0

    # This variable will store the response from the client.chat.completions.create call
    api_response = None
    while attempt < max_retries:
        try:
            api_response = client.chat.completions.create(
                model=model,
                messages=messages,
                **sample_params,
                extra_body=extra_body,
            )
            # If the call is successful, break out of the loop
            break
        except Exception as e:
            attempt += 1
            logging.warning(
                f"API call in send_request failed on attempt {attempt}/{max_retries}. Error: {e}"
            )
            if attempt < max_retries:
                logging.info(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                logging.error(
                    f"All {max_retries} API call attempts in send_request failed. Last error: {e}"
                )
                # api_response will remain None if all retries fail

    # Return the API response (which could be None if all retries failed) and the messages
    return api_response, messages


def solve_problem(
    problem_id,
    problem_text,
    template,
    thinking,
    thinking_interven,
    client_instance,
    llm_model_name,
    expected_answer_str,
):
    template_str = ""
    template_prompt = ""
    if template:
        template_prompt = "You are given a template to solve the problem. Use the given steps if applicable otherwise use them to guide your reasoning and present the solution steps logically to solve this problem:\n"
        # check if "Step i" is in the template otherwise use the default numbering
        for i, step in enumerate(template):
            step = step.strip()
            if re.match(r"^Step \d+:", step):
                template_str += f"{step}\n"
            elif re.match(r"^\d+\.", step):
                # Replace the numbered prefix like "1." with "Step 1:"
                template_str += f"{re.sub(r'^\d+\.', f'Step {i + 1}:', step, 1)}\n"
            else:
                template_str += f"Step {i + 1}: {step}\n"

    prompt_base_instructions = """Solve the following math problem efficiently and clearly. Present the solution steps logically.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final numerical answer that solves the problem. Ensure the number is clearly identifiable within the box."""

    send_request_thinking_interven = False
    if thinking_interven and template_str:
        send_request_thinking_interven = True
        current_prompt = f"{template_prompt}\nProblem: {problem_text}.<|im_end|>\n<|im_start|>assistant<|im_sep|>\n<think>I need to follow the given steps to guide me in reasoning to solve the problem. I will follow each step and modify the steps to make them match the problem and then solve the problem accordingly. The template is. {template_str.replace('\n', ' ')}Strictly following the given steps for guidance, I will now solve the problem starting from the step 1. Using step 1"
        logging.info(
            f"Processing problem {problem_id} with template and thinking intervention."
        )
    elif template_str:
        current_prompt = f"{template_prompt}{template_str}\n\nProblem: {problem_text}"
        logging.info(f"Processing problem {problem_id} with template.")
    else:
        current_prompt = f"{prompt_base_instructions}\n\nProblem: {problem_text}\nLet's solve this step-by-step "
        logging.info(f"Processing problem {problem_id} without template.")

    # logging.info(f"Current prompt for problem {problem_id}:\n{current_prompt}")
    response, messages = send_request(
        current_prompt,
        client_instance,
        llm_model_name,
        enable_thinking=thinking,
        thinking_interven=send_request_thinking_interven,
    )
    time.sleep(2)
    if response is None:
        logging.error(
            f"Failed to get a response from the model for problem {problem_id}. Exiting."
        )
        return None, None, None, None, None

    message = response.choices[0].message.content
    think_content = response.choices[0].message.reasoning_content
    no_think_message = ""

    if message:
        output_tokens_count = response.usage.completion_tokens
        output_text = think_content + message
    else:
        output_text = think_content
        output_tokens_count = response.usage.completion_tokens
    # elif thinking:
    #     logging.info(f"Re-Sending a no-think request with cut think content due to MAX THINKING TOKEN limit")
    #     think_msg = {
    #         "role": "assistant",
    #         "content": "<think>" + think_content + "\n</think>\n\n",
    #     }
    #     messages.append(think_msg)
    #     no_think_response, messages_think = send_request(
    #         current_prompt,
    #         client_instance,
    #         llm_model_name,
    #         enable_thinking=False,
    #         think_content=messages,
    #     )
    #     no_think_resp_think = no_think_response.choices[0].message.reasoning_content
    #     no_think_message = no_think_response.choices[0].message.content
    #     if no_think_message is None:
    #         output_text = think_content + no_think_resp_think
    #     else:
    #         output_text = think_content + no_think_resp_think + no_think_message
    #     output_tokens_count = (
    #         response.usage.completion_tokens + no_think_response.usage.completion_tokens
    #     )

    extracted_answer_str, is_correct = extract_and_validate_answer(
        output_text, expected_answer_str, problem_id
    )

    return output_text, output_tokens_count, extracted_answer_str, messages, is_correct


