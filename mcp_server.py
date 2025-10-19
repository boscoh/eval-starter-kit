#!/usr/bin/env python3
"""
MCP Server for xConf Assistant - Speaker Recommendation System

This server exposes the get_best_speaker function as an MCP tool,
allowing AI agents to find the most relevant speaker for a given query.
"""

import copy
import logging
import sys
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from rag import RAGService
from setup_logger import setup_logging_with_rich_logger

setup_logging_with_rich_logger(level=logging.INFO)


logger = logging.getLogger(__name__)

mcp = FastMCP("Simle MCP")

rag_service = RAGService("bedrock")


async def ainit():
    global rag_service
    try:
        await rag_service.get_speakers()
        logger.info(f"Initialized with {len(rag_service.speakers)} speakers")
    except Exception as e:
        logger.error(f"Error initializing data: {e}")
        raise


@mcp.tool()
async def get_best_speaker(query: str) -> Dict[str, Any]:
    """
    Find the most relevant speaker for a given topic or query using semantic similarity.

    This tool uses AI-powered semantic search to match your query against speaker bios,
    abstracts, and expertise areas to find the best match. Perfect for finding speakers
    who can speak on specific topics, technologies, or subject areas.

    Args:
        query: A description of the topic, technology, or expertise area you need a speaker for

    Returns:
        Dict containing the best matching speaker with their bio, abstract, and relevance details
    """
    if not rag_service or not rag_service.speakers:
        return {
            "error": "No speaker data available. Please ensure storage.json exists."
        }
    try:
        best_speaker = await rag_service.get_best_speaker(query)
        return {
            "success": True,
            "speaker": best_speaker,
            "query": query,
            "total_speakers_searched": len(rag_service.speakers),
        }
    except Exception as e:
        logger.error(f"Error in get_best_speaker: {e}")
        return {"success": False, "error": str(e), "query": query}


@mcp.tool()
async def get_speaker_by_name(name: str) -> Dict[str, Any]:
    """
    Retrieve detailed information about a specific speaker by their exact name.

    Use this tool when you know the speaker's name and want to get their complete
    profile including bio, abstract, company, and title. This is useful for getting
    detailed information about a speaker you've already identified.

    Args:
        name: The exact name of the speaker to find (case-insensitive)

    Returns:
        Dict containing the speaker's complete information if found, or error if not found
    """
    if not rag_service or not rag_service.speakers:
        return {
            "error": "No speaker data available. Please ensure storage.json exists."
        }

    try:
        for speaker in await rag_service.get_speakers():
            if speaker.get("name", "").lower() == name.lower():
                return {"success": True, "speaker": speaker}

        return {
            "success": False,
            "error": f"Speaker '{name}' not found",
            "available_speakers": [s.get("name", "Unknown") for s in rag_service.speakers[:10]],
        }

    except Exception as e:
        logger.error(f"Error in get_speaker_by_name: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_all_speakers() -> Dict[str, Any]:
    """
    Get a comprehensive list of all available speakers with their basic information.

    This tool provides an overview of all speakers in the database, including their
    names, titles, companies, and brief descriptions. Use this to browse available
    speakers or get a sense of the speaker pool before searching for specific topics.

    Returns:
        Dict containing a list of all speakers with their basic profile information
    """
    try:
        speakers_list = []
        for speaker in await rag_service.get_speakers():
            clean_speaker = {
                "name": speaker.get("name", "Unknown"),
                "title": speaker.get("title", "Unknown"),
                "company": speaker.get("company", "Unknown"),
                "bio_max_120_words": speaker.get("bio_max_120_words", ""),
                "final_abstract_max_150_words": speaker.get(
                    "final_abstract_max_150_words", ""
                ),
            }
            speakers_list.append(clean_speaker)

        return {
            "success": True,
            "speakers": speakers_list,
            "intro_message": "**Conference Speakers from the data**",
            "total_count": len(speakers_list),
        }

    except Exception as e:
        logger.error(f"Error in list_all_speakers: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Main function to run the MCP server."""
    import asyncio

    try:
        asyncio.run(ainit())
        logger.info("Starting MCP Server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
