import re
import cv2
import io
import base64
import sympy as sp
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple, Dict, Any


class EquationProcessor:
    def __init__(self):
        # Common physics symbols and their LaTeX representations
        self.symbol_mapping = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'theta': 'θ', 'lambda': 'λ', 'mu': 'μ', 'pi': 'π',
            'sigma': 'σ', 'omega': 'ω', 'phi': 'φ', 'psi': 'ψ',
            'nabla': '∇', 'partial': '∂', 'infinity': '∞',
            'hbar': 'ℏ', 'dot': '·', 'ddot': '··',
            # Add more mathematical symbols
            'times': '×', 'div': '÷', 'pm': '±', 'mp': '∓',
            'leq': '≤', 'geq': '≥', 'neq': '≠', 'approx': '≈',
            'propto': '∝', 'in': '∈', 'notin': '∉', 'subset': '⊂',
            'supset': '⊃', 'cup': '∪', 'cap': '∩', 'emptyset': '∅'
        }
        
        # Common physics operators
        self.operators = {
            'derivative': ['d/dx', '∂/∂x', '∇', 'd²/dx²', '∂²/∂x²'],
            'integral': ['∫', '∮', '∬', '∭', '∫∫', '∫∫∫'],
            'sum': ['∑', '∏', '∐'],
            'vector': ['→', '←', '↔', '↑', '↓'],
            'set': ['∈', '∉', '⊂', '⊃', '∪', '∩', '∅'],
            'logic': ['∀', '∃', '∄', '∧', '∨', '¬', '⇒', '⇔']
        }
        
        self.latex_patterns = {
            'fraction': r'\\frac\{([^}]+)\}\{([^}]+)\}',
            'square_root': r'\\sqrt\{([^}]+)\}',
            'power': r'\^\{([^}]+)\}',
            'subscript': r'_\{([^}]+)\}'
        }
    
    def process_equation_image(self, image_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Process an equation from an image.
        Returns the equation in LaTeX format and metadata.
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        print(f'image: {image}')
        # Preprocess image for better OCR
        # processed_image = self._preprocess_image(image)
        
        # Extract equation using OCR with improved configuration
        equation_text = pytesseract.image_to_string(
            image)
        print(f'equation_text: {equation_text}')
        # Convert to LaTeX format
        latex_equation = self._convert_to_latex(equation_text)
        
        # Extract metadata
        metadata = self._extract_equation_metadata(latex_equation)
        
        return latex_equation, metadata
    
    def process_equation_text(self, equation_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process an equation from text input.
        Returns the equation in LaTeX format and metadata.
        """
        # Convert to LaTeX format
        latex_equation = self._convert_to_latex(equation_text)
        
        # Extract metadata
        metadata = self._extract_equation_metadata(latex_equation)
        
        return latex_equation, metadata
    
    # def process_base64_image(self, image_data: str) -> Tuple[str, Dict[str, Any]]:
    #     """
    #     Process a base64 encoded image containing a physics equation.
        
    #     Args:
    #         image_data: Base64 encoded image string
            
    #     Returns:
    #         Tuple[str, Dict[str, Any]]: Processed equation in LaTeX format and metadata
    #     """
    #     # TODO: Implement actual image processing and OCR
    #     # For now, return a placeholder equation
    #     return "E = mc^2", {"confidence": 1.0}
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(denoised)
        
        return enhanced
    
    def _convert_to_latex(self, equation_text: str) -> str:
        """
        Convert equation text to LaTeX format.
        """
        # Replace common physics symbols
        for symbol, latex in self.symbol_mapping.items():
            equation_text = equation_text.replace(symbol, latex)
        
        # Handle fractions
        equation_text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', equation_text)
        
        # Handle exponents
        equation_text = re.sub(r'(\w+)\^(\w+)', r'\1^{\2}', equation_text)
        
        # Handle subscripts
        equation_text = re.sub(r'(\w+)_(\w+)', r'\1_{\2}', equation_text)
        
        # Handle square roots
        equation_text = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', equation_text)
        
        # Handle integrals
        equation_text = re.sub(r'int\(([^)]+)\)', r'\\int \1', equation_text)
        
        # Handle summations
        equation_text = re.sub(r'sum\(([^)]+)\)', r'\\sum \1', equation_text)
        
        # Handle products
        equation_text = re.sub(r'prod\(([^)]+)\)', r'\\prod \1', equation_text)
        
        return equation_text
    
    def _extract_equation_metadata(self, latex_equation: str) -> Dict[str, Any]:
        """
        Extract metadata from the equation.
        """
        metadata = {
            'type': self._determine_equation_type(latex_equation),
            'variables': self._extract_variables(latex_equation),
            'operators': self._extract_operators(latex_equation),
            'complexity': self._calculate_complexity(latex_equation),
            'domain': self._determine_equation_domain(latex_equation)
        }
        return metadata
    
    def _determine_equation_type(self, equation: str) -> str:
        """
        Determine the type of equation (e.g., differential, integral, algebraic).
        """
        if any(op in equation for op in self.operators['derivative']):
            return 'differential'
        elif any(op in equation for op in self.operators['integral']):
            return 'integral'
        elif any(op in equation for op in self.operators['sum']):
            return 'summation'
        elif any(op in equation for op in self.operators['vector']):
            return 'vector'
        else:
            return 'algebraic'
    
    def _determine_equation_domain(self, equation: str) -> str:
        """
        Determine the domain of the equation (e.g., mechanics, electromagnetism).
        """
        mechanics_keywords = ['force', 'mass', 'velocity', 'acceleration', 'momentum', 'energy']
        em_keywords = ['electric', 'magnetic', 'field', 'charge', 'current', 'voltage']
        quantum_keywords = ['wave', 'particle', 'quantum', 'spin', 'state', 'operator']
        
        equation_lower = equation.lower()
        
        if any(keyword in equation_lower for keyword in mechanics_keywords):
            return 'mechanics'
        elif any(keyword in equation_lower for keyword in em_keywords):
            return 'electromagnetism'
        elif any(keyword in equation_lower for keyword in quantum_keywords):
            return 'quantum'
        else:
            return 'general'
    
    def _extract_variables(self, equation: str) -> list:
        """
        Extract variables from the equation.
        """
        # Use sympy to parse the equation
        try:
            expr = sp.sympify(equation)
            return list(expr.free_symbols)
        except:
            # Fallback to regex if sympy fails
            return re.findall(r'[a-zA-Z][a-zA-Z0-9]*', equation)
    
    def _extract_operators(self, equation: str) -> list:
        """
        Extract operators from the equation.
        """
        operators = []
        for op_type, op_list in self.operators.items():
            for op in op_list:
                if op in equation:
                    operators.append(op)
        return operators
    
    def _calculate_complexity(self, equation: str) -> int:
        """
        Calculate the complexity score of the equation.
        """
        complexity = 0
        
        # Count operators
        for op_list in self.operators.values():
            complexity += sum(equation.count(op) for op in op_list)
        
        # Count fractions
        complexity += equation.count('\\frac') * 2
        
        # Count exponents and subscripts
        complexity += equation.count('^') + equation.count('_')
        
        # Count parentheses
        complexity += equation.count('(') + equation.count(')')
        
        # Count special functions
        complexity += equation.count('\\sqrt') * 2
        complexity += equation.count('\\int') * 3
        complexity += equation.count('\\sum') * 2
        complexity += equation.count('\\prod') * 2
        
        return complexity
    
    def validate_latex(self, latex: str) -> bool:
        """
        Validate if a string is valid LaTeX.
        
        Args:
            latex: LaTeX string to validate
            
        Returns:
            bool: True if valid LaTeX, False otherwise
        """
        # Basic validation - check for common LaTeX patterns
        for pattern in self.latex_patterns.values():
            if re.search(pattern, latex):
                return True
        return False
    
    def extract_equations(self, text: str) -> list:
        """
        Extract equations from text.
        
        Args:
            text: Text containing equations
            
        Returns:
            list: List of extracted equations
        """
        # Find equations between $ or $$ delimiters
        equations = re.findall(r'\$([^$]+)\$|\$\$([^$]+)\$\$', text)
        return [eq[0] or eq[1] for eq in equations]

# Create a singleton instance
equation_processor = EquationProcessor() 