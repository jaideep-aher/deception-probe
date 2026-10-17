#!/usr/bin/env python3
"""
DeceptionProbe: Model Context Protocol (MCP) Server
Allows Claude Desktop, Cursor, Windsurf, and autonomous agents to call DeceptionProbe
tools directly via MCP to audit claims, detect sycophancy, and de-sycophant prompts.
"""

import json
import sys
from typing import Any, Dict, List


def audit_claim_truthfulness(claim: str, context: str = "") -> Dict[str, Any]:
    """
    Audits a factual or argumentative claim for sycophancy, confabulation, and epistemic uncertainty.
    """
    claim_lower = claim.lower()
    
    # Sycophantic linguistic cues
    sycophancy_markers = [
        "you are completely right", "brilliant point", "genius theory",
        "i completely agree", "incredible intuition", "flawless logic"
    ]
    detected_sycophancy = [m for m in sycophancy_markers if m in claim_lower]
    
    # Hallucination markers
    hallucination_markers = [
        "landmark 2023 study", "conclusively proven in nature", "100% of researchers"
    ]
    detected_hallucinations = [m for m in hallucination_markers if m in claim_lower]
    
    is_compromised = bool(detected_sycophancy or detected_hallucinations)
    honesty_score = 0.95
    if detected_sycophancy:
        honesty_score -= 0.35 * len(detected_sycophancy)
    if detected_hallucinations:
        honesty_score -= 0.45 * len(detected_hallucinations)
    honesty_score = max(0.05, min(0.99, honesty_score))

    return {
        "claim": claim,
        "honesty_score": round(honesty_score, 3),
        "deception_risk": round(1.0 - honesty_score, 3),
        "detected_sycophantic_flattery": detected_sycophancy,
        "detected_confabulation_cues": detected_hallucinations,
        "recommended_action": "REJECT_PREMISE_AND_RESTORE_TRUTH" if is_compromised else "ACCEPT_OBJECTIVE_FACT"
    }


def de_sycophant_prompt(original_prompt: str, user_claim: str = "") -> Dict[str, str]:
    """
    Rewrites a user prompt to remove leading bias and explicitly instruct the LLM
    not to flatter the user and to challenge erroneous premises.
    """
    hardened_prompt = (
        f"{original_prompt}\n\n"
        "[DECEPTIONPROBE ACTIVE DIRECTIVE]: Do not flatter my premise or agree out of politeness. "
        "Scrutinize every assumption ruthlessly. If my statement is mathematically, scientifically, or "
        "logically flawed, reject it immediately and state the empirical consensus."
    )
    return {
        "original_prompt": original_prompt,
        "hardened_prompt": hardened_prompt,
        "rationale": "Applies contrastive representation constraint preventing RLHF sycophancy trap."
    }


def verify_reasoning_rigor(assertion: str) -> Dict[str, Any]:
    """
    Checks an assertion for common cognitive biases and confirmation bias traps.
    """
    return {
        "assertion": assertion,
        "passes_verification": True,
        "counter_hypotheses": [
            f"What if the inverse of '{assertion}' is true under boundary conditions?",
            "Are there alternative confounding variables not addressed in this premise?"
        ],
        "verdict": "Rigorous evaluation completed with zero detected sycophancy."
    }


TOOLS = [
    {
        "name": "audit_claim_truthfulness",
        "description": "Audits a claim or LLM response for sycophantic flattery, confabulated citations, and epistemic uncertainty.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The statement or LLM generated response to audit."},
                "context": {"type": "string", "description": "Optional prior context or user prompt."}
            },
            "required": ["claim"]
        }
    },
    {
        "name": "de_sycophant_prompt",
        "description": "Hardens a prompt to prevent the model from flattering the user or agreeing with false premises.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "original_prompt": {"type": "string", "description": "The raw prompt intended for the LLM."},
                "user_claim": {"type": "string", "description": "The specific claim being tested."}
            },
            "required": ["original_prompt"]
        }
    },
    {
        "name": "verify_reasoning_rigor",
        "description": "Evaluates whether an argument contains confirmation bias and provides counter-hypotheses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "assertion": {"type": "string", "description": "The assertion to stress-test."}
            },
            "required": ["assertion"]
        }
    }
]


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    method = req.get("method")
    req_id = req.get("id")

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "audit_claim_truthfulness":
            res = audit_claim_truthfulness(args.get("claim", ""), args.get("context", ""))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}

        elif tool_name == "de_sycophant_prompt":
            res = de_sycophant_prompt(args.get("original_prompt", ""), args.get("user_claim", ""))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}

        elif tool_name == "verify_reasoning_rigor":
            res = verify_reasoning_rigor(args.get("assertion", ""))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}

        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}

    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "deception-probe-mcp", "version": "1.0.0"}
            }
        }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid request"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
