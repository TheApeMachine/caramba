package tools

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"strconv"
)

// Calculator is a tool for performing calculations
type Calculator struct{}

// NewCalculator creates a new Calculator tool
func NewCalculator() *Calculator {
	return &Calculator{}
}

// Name returns the name of the tool
func (c *Calculator) Name() string {
	return "calculator"
}

// Description returns the description of the tool
func (c *Calculator) Description() string {
	return "Performs mathematical calculations"
}

// Execute executes the tool with the given arguments
func (c *Calculator) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	expression, ok := args["expression"].(string)
	if !ok {
		return nil, errors.New("expression must be a string")
	}
	
	result, err := c.evaluate(expression)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate expression: %w", err)
	}
	
	return result, nil
}

// Schema returns the JSON schema for the tool's arguments
func (c *Calculator) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"expression": map[string]interface{}{
				"type":        "string",
				"description": "The mathematical expression to evaluate",
			},
		},
		"required": []string{"expression"},
	}
}

// evaluate evaluates a mathematical expression
func (c *Calculator) evaluate(expr string) (float64, error) {
	// Parse the expression
	tree, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("failed to parse expression: %w", err)
	}
	
	// Evaluate the expression
	return c.evaluateAst(tree)
}

// evaluateAst evaluates an AST node
func (c *Calculator) evaluateAst(node ast.Expr) (float64, error) {
	switch n := node.(type) {
	case *ast.BasicLit:
		// Handle basic literals (numbers)
		if n.Kind == token.INT || n.Kind == token.FLOAT {
			return strconv.ParseFloat(n.Value, 64)
		}
		return 0, fmt.Errorf("unsupported literal type: %v", n.Kind)
		
	case *ast.BinaryExpr:
		// Handle binary expressions (e.g., a + b)
		x, err := c.evaluateAst(n.X)
		if err != nil {
			return 0, err
		}
		
		y, err := c.evaluateAst(n.Y)
		if err != nil {
			return 0, err
		}
		
		switch n.Op {
		case token.ADD:
			return x + y, nil
		case token.SUB:
			return x - y, nil
		case token.MUL:
			return x * y, nil
		case token.QUO:
			if y == 0 {
				return 0, errors.New("division by zero")
			}
			return x / y, nil
		case token.REM:
			return float64(int(x) % int(y)), nil
		default:
			return 0, fmt.Errorf("unsupported operator: %v", n.Op)
		}
		
	case *ast.ParenExpr:
		// Handle parenthesized expressions
		return c.evaluateAst(n.X)
		
	case *ast.UnaryExpr:
		// Handle unary expressions (e.g., -a)
		x, err := c.evaluateAst(n.X)
		if err != nil {
			return 0, err
		}
		
		switch n.Op {
		case token.SUB:
			return -x, nil
		case token.ADD:
			return x, nil
		default:
			return 0, fmt.Errorf("unsupported unary operator: %v", n.Op)
		}
		
	case *ast.CallExpr:
		// Handle function calls (e.g., sqrt(a))
		if ident, ok := n.Fun.(*ast.Ident); ok {
			// Get the function arguments
			args := make([]float64, len(n.Args))
			for i, arg := range n.Args {
				val, err := c.evaluateAst(arg)
				if err != nil {
					return 0, err
				}
				args[i] = val
			}
			
			// Execute the function
			switch ident.Name {
			case "sqrt":
				if len(args) != 1 {
					return 0, errors.New("sqrt requires exactly one argument")
				}
				return math.Sqrt(args[0]), nil
			case "sin":
				if len(args) != 1 {
					return 0, errors.New("sin requires exactly one argument")
				}
				return math.Sin(args[0]), nil
			case "cos":
				if len(args) != 1 {
					return 0, errors.New("cos requires exactly one argument")
				}
				return math.Cos(args[0]), nil
			case "tan":
				if len(args) != 1 {
					return 0, errors.New("tan requires exactly one argument")
				}
				return math.Tan(args[0]), nil
			case "exp":
				if len(args) != 1 {
					return 0, errors.New("exp requires exactly one argument")
				}
				return math.Exp(args[0]), nil
			case "log":
				if len(args) != 1 {
					return 0, errors.New("log requires exactly one argument")
				}
				return math.Log(args[0]), nil
			case "pow":
				if len(args) != 2 {
					return 0, errors.New("pow requires exactly two arguments")
				}
				return math.Pow(args[0], args[1]), nil
			default:
				return 0, fmt.Errorf("unknown function: %s", ident.Name)
			}
		}
		return 0, errors.New("invalid function call")
		
	default:
		return 0, fmt.Errorf("unsupported expression type: %T", node)
	}
}
