from agentai.agent import ToxicityAgent
import sys

def main():
    print("\n" + "="*60)
    print("🛡️  TOXICITY DETECTION SYSTEM")
    print("="*60)
    print("\nThis system analyzes content for toxicity and provides")
    print("constructive feedback to help improve online communication.")
    print("="*60 + "\n")
    
    # Initialize agent
    try:
        agent = ToxicityAgent()
    except FileNotFoundError:
        print(" Error: Toxicity examples not found!")
        print("\nPlease run the preprocessing step first:")
        print("  python preprocess_data.py")
        return
    except RuntimeError as e:
        print(f"  {e}")
        return
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("1. Analyze single content")
        print("2. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            # Single analysis
            print("\nEnter the content to analyze (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            
            content = "\n".join(lines).strip()
            
            if content.strip():
                result = agent.detect_and_respond(content)
                agent.display_result(result)
            else:
                print("⚠️  No content entered.")
            
        elif choice == '2':
            print("\n Thank you for using the Toxicity Detection System!")
            print("="*60 + "\n")
            break
        
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!")
        sys.exit(0)