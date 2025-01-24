# Testing the Cognitive Flexibility Experiments

## Prerequisites

Before running tests, ensure you have:
- Python 3.8+
- All project dependencies installed (`pip install -r requirements.txt`)

## Running Tests

### Automated Test Suite

To run all tests:

```bash
python -m unittest discover tests
```

### Specific Test Modules

Run specific test modules:

```bash
# Integration tests
python -m unittest tests.test_integration

# Configuration tests 
python -m unittest tests.test_config
```

## Test Coverage

The test suite covers:

### WCST (Wisconsin Card Sorting Test)
- Initialization
- Option generation
- Rule switching mechanism
- Performance tracking

### LNT (Letter Number Test)
- Sequence generation
- Task switching
- Response evaluation

### Configuration Utilities
- Config file loading
- Model configuration extraction
- Configuration validation

### Logging
- Logger setup and functionality

### Mock Model
- Simulated model interactions

## Adding New Tests

1. Create test files in the `tests/` directory
2. Use `unittest` framework
3. Follow existing test structure
4. Add comprehensive test cases

## Troubleshooting

- Ensure all dependencies are installed
- Check Python version compatibility
- Verify configuration files are present in `config/` directory

## Notes

- Tests use mock objects to simulate model interactions
- Some tests may require network access for API-related checks