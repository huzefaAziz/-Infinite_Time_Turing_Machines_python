# Infinite Time Turing Machine Implementation

A Python implementation of Infinite Time Turing Machines (ITTMs) based on the paper by Joel David Hamkins and Andy Lewis: [Infinite Time Turing Machines](https://arxiv.org/pdf/math/9808093).

## Overview

Infinite Time Turing Machines extend classical Turing machines to operate in transfinite ordinal time, allowing for supertasks - computations involving infinitely many steps. This implementation provides:

- **Three-tape architecture**: Input, scratch, and output tapes
- **Limit ordinal behavior**: Proper handling of limit stages with limsup values
- **Supertask computation**: Ability to compute beyond finite time

## Key Features

### Core Components

1. **Tape Class**: Represents infinite tapes (input, scratch, output) that can store binary values
2. **InfiniteTimeTuringMachine Class**: The main machine implementation
3. **Snapshot System**: Tracks machine state at different times
4. **Limit Behavior**: Implements the limsup rule for limit ordinal times

### How ITTMs Work

1. **Finite Steps**: Like classical Turing machines, the machine executes transitions based on:
   - Current state
   - Values under the head on all three tapes
   - Transition table (program)

2. **Limit Ordinal Times**: At limit ordinal times:
   - Head moves to position 0 (leftmost)
   - State becomes the special LIMIT state
   - Each cell takes the limsup value:
     - If the cell changed infinitely often before the limit → becomes 1
     - Otherwise → takes the final value it had

3. **After Limits**: The machine continues computation from the limit state

## Usage

```python
from infinite_time_turing_machine import InfiniteTimeTuringMachine, create_copy_program

# Create a machine with a program
machine = InfiniteTimeTuringMachine(create_copy_program())
machine.set_input("1011")

# Run the machine
machine.run(max_steps=100, verbose=True)

# Get output
output = machine.get_output()
print(f"Output: {output}")
```

## Program Format

Programs are dictionaries mapping:
```
(state, input_bit, scratch_bit, output_bit) → 
(new_state, new_input_bit, new_scratch_bit, new_output_bit, direction)
```

Where:
- `state`: Current machine state (string)
- `input_bit`, `scratch_bit`, `output_bit`: Values (0 or 1) under the head
- `new_state`: Next state
- `new_input_bit`, etc.: Values to write
- `direction`: 'L' (left), 'R' (right), or 'S' (stay)

## Example Programs

The implementation includes example programs:
- `create_copy_program()`: Copies input to output
- `create_increment_program()`: Increments a binary number

## Mathematical Background

According to the paper:
- Every Π₁¹ set is decidable by ITTMs
- Semi-decidable sets form a portion of Δ₂¹ sets
- The theory leads to rich degree structures and jump operators
- Clockable and writable ordinals have interesting properties

## Limitations

This implementation focuses on the core mechanics. For full theoretical power:
- Proper limit ordinal handling requires more sophisticated tracking
- Infinite change detection needs careful history management
- Real ITTM computations can run for very long (transfinite) times

## References

- Hamkins, J. D., & Lewis, A. (2000). Infinite Time Turing Machines. *Journal of Symbolic Logic*, 65(2), 567-604. [arXiv:math/9808093](https://arxiv.org/pdf/math/9808093)

## License

This is an educational implementation based on the research paper.

