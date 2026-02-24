from rag_setup import ToxicityRAG
import re

class ToxicityAgent:
    def __init__(self):
        """Initialize the Toxicity Detection Agent"""
        print("\n" + "="*60)
        print("TOXICITY DETECTION AGENT")
        print("="*60)
        
        self.rag = ToxicityRAG()
        
        print("="*60)
        print("Agent is ready to analyze content!")
        print("="*60 + "\n")
    
    def detect_and_respond(self, content):
        """
        Detect toxicity and generate response
        
        Args:
            content (str): The content to analyze
            
        Returns:
            dict: Analysis results with classification, explanation, and message
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING CONTENT")
        print(f"{'='*60}")
        print(f"Content: {content[:150]}{'...' if len(content) > 150 else ''}")
        print(f"{'='*60}\n")
        
        # Get analysis from RAG
        print("Running analysis...")
        analysis = self.rag.analyze_content(content)
        
        # Parse result
        result = self._parse_analysis(analysis)
        
        return result
    
    def _parse_analysis(self, analysis):
        """Parse LLM response into structured format"""
        result = {
            'classification': 'UNKNOWN',
            'explanation': '',
            'message_to_author': '',
            'raw_response': analysis
        }
        
        lines = analysis.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Parse classification
            if line.startswith('Classification:'):
                classification = line.replace('Classification:', '').strip()
                # Extract just the classification word
                match = re.search(r'(TOXIC|NEUTRAL|GOOD)', classification, re.IGNORECASE)
                if match:
                    result['classification'] = match.group(1).upper()
            
            # Parse explanation
            elif line.startswith('Explanation:'):
                current_section = 'explanation'
                result['explanation'] = line.replace('Explanation:', '').strip()
            
            # Parse message
            elif line.startswith('Message to author:'):
                current_section = 'message'
                result['message_to_author'] = line.replace('Message to author:', '').strip()
            
            # Continue building sections
            elif line and current_section == 'explanation' and not line.startswith('Message'):
                result['explanation'] += ' ' + line
            elif line and current_section == 'message':
                result['message_to_author'] += ' ' + line
        
        return result
    
    def display_result(self, result):
        """Display analysis results in formatted output"""
        classification = result['classification']
        
        # Color codes for terminal
        colors = {
            'TOXIC': '\033[91m',     # Red
            'NEUTRAL': '\033[93m',   # Yellow
            'GOOD': '\033[92m',      # Green
            'UNKNOWN': '\033[0m'     # Default
        }
        reset = '\033[0m'
        
        # Emojis
        emojis = {
            'TOXIC': '',
            'NEUTRAL': '',
            'GOOD': '',
            'UNKNOWN': ''
        }
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*60}\n")
        
        emoji = emojis.get(classification, '')
        color = colors.get(classification, colors['UNKNOWN'])
        
        print(f"{emoji} {color}Classification: {classification}{reset}\n")
        print(f"Explanation:")
        print(f"   {result['explanation']}\n")
        
        if result['message_to_author'] and result['message_to_author'] != 'N/A':
            print(f"Message to Author:")
            print(f"   {result['message_to_author']}\n")
        
        print(f"{'='*60}\n")
    
    def batch_analyze(self, contents):
        """Analyze multiple contents at once"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"📦 BATCH ANALYSIS: {len(contents)} items")
        print(f"{'='*60}\n")
        
        for i, content in enumerate(contents, 1):
            print(f"[{i}/{len(contents)}] Analyzing...")
            result = self.detect_and_respond(content)
            results.append(result)
        
        # Summary
        classifications = [r['classification'] for r in results]
        print(f"\n{'='*60}")
        print(f"BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"TOXIC: {classifications.count('TOXIC')}")
        print(f"NEUTRAL: {classifications.count('NEUTRAL')}")
        print(f"GOOD: {classifications.count('GOOD')}")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    agent = ToxicityAgent()
    
    test_cases = [
        "You're a complete idiot and nobody likes you.",
        "I respectfully disagree with your point about climate change.",
        "This is amazing work! Thank you for your contribution!",
        "FUCK OFF YOU PIECE OF SHIT",
        "Can you provide more details about your methodology?"
    ]
    
    print("\nRunning test cases...\n")
    
    for content in test_cases:
        result = agent.detect_and_respond(content)
        agent.display_result(result)