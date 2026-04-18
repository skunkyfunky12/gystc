import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        print("Index command not yet implemented.", file=sys.stderr)
        sys.exit(1)
    from brain_mcp.server import mcp
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
