import argparse
from graph import build_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous content pipeline")
    parser.add_argument("topic", help="Topic for the blog post")
    args = parser.parse_args()

    app = build_app()
    final = app.invoke({"topic": args.topic})
    print(f"\nPublished: {final['output_path']}")
    print(f"Loops: {final['loops']} | Score: {final['score']:.2f} | Flagged: {final['flagged']}")