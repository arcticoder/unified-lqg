#!/usr/bin/env python3
"""
Quick script to fix syntax warnings in lqg_fixed_components.py
"""

def fix_syntax_warnings():
    with open("lqg_fixed_components.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix the two problematic docstrings by converting to raw strings
    fixes = [
        # First occurrence around line 425
        ('    def kx_operator(self, site: int) -> sp.csr_matrix:\n        """\n        Build \\widehat{K}_x(site)',
         '    def kx_operator(self, site: int) -> sp.csr_matrix:\n        r"""\n        Build \\widehat{K}_x(site)'),
        
        # Second occurrence around line 448
        ('    def kphi_operator(self, site: int) -> sp.csr_matrix:\n        """\n        Build \\widehat{K}_φ(site)',
         '    def kphi_operator(self, site: int) -> sp.csr_matrix:\n        r"""\n        Build \\widehat{K}_φ(site)')
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    with open("lqg_fixed_components.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✓ Fixed syntax warnings by converting docstrings to raw strings")

if __name__ == "__main__":
    fix_syntax_warnings()
