from agent import ToxicityAgent
import sys

def main():
    """Main interface for the toxicity detection system"""
    
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
        print("❌ Error: Toxicity examples not found!")
        print("\nPlease run the preprocessing step first:")
        print("  python preprocess_data.py")
        return
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # Interactive mode
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("1. Analyze single content")
        print("2. Run test examples")
        print("3. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Single analysis
            print("\nEnter the content to analyze (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            
            content = "\n".join(lines)
            
            if content.strip():
                result = agent.detect_and_respond(content)
                agent.display_result(result)
                
                # Ask if user wants to see the message
                if result['classification'] == 'TOXIC':
                    send = input("\n📤 Send this message to the content author? (y/n): ").strip().lower()
                    if send == 'y':
                        print("\n✅ Message sent (simulated):")
                        print(f"   {result['message_to_author']}")
            else:
                print("⚠️  No content entered.")
        
        elif choice == '2':
            # Test examples
            test_cases = [
                "You're such a fucking idiot. Go kill yourself.",
                "I disagree with your approach, but I respect your perspective.",
                "This is incredible work! Thank you so much for sharing!",
                "Women are too emotional to be leaders.",
                "Can you clarify what you mean by that statement?"
            ]
            
            print(f"\n🧪 Running {len(test_cases)} test cases...\n")
            
            results = agent.batch_analyze(test_cases)
            
            for i, result in enumerate(results, 1):
                print(f"\nTest Case {i}:")
                agent.display_result(result)
        
        elif choice == '3':
            print("\n👋 Thank you for using the Toxicity Detection System!")
            print("="*60 + "\n")
            break
        
        else:
            print("⚠️  Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)