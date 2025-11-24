import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from lang-chain!")
    print(f"GoogleApi : {os.environ.get("GOOGLE_API_KEY")}")


if __name__ == "__main__":
    main()
