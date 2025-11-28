"""
Example programs and demonstrations for Infinite Time Turing Machines
"""

from infinite_time_turing_machine import InfiniteTimeTuringMachine, Tape, State


def example_binary_addition():
    """
    Example: Add two binary numbers.
    This is a simplified version - full implementation would be more complex.
    """
    print("=" * 60)
    print("Example: Binary Addition (Simplified)")
    print("=" * 60)
    
    # This is a placeholder - a full binary addition program would be quite complex
    # For demonstration, we'll create a simple program that processes bits
    program = {
        # Start: initialize
        ('start', 0, 0, 0): ('process', 0, 0, 0, 'R'),
        ('start', 1, 0, 0): ('process', 1, 0, 0, 'R'),
        
        # Process bits (simplified)
        ('process', 0, 0, 0): ('process', 0, 0, 0, 'R'),
        ('process', 1, 0, 0): ('process', 1, 0, 1, 'R'),
    }
    
    machine = InfiniteTimeTuringMachine(program)
    machine.set_input("101")  # Binary 5
    
    print(f"Input: {machine.input_tape.get_content_string()}")
    machine.run(max_steps=10, verbose=True)
    print(f"Output: {machine.get_output()}")


def example_tape_operations():
    """Demonstrate basic tape operations"""
    print("\n" + "=" * 60)
    print("Example: Tape Operations")
    print("=" * 60)
    
    tape = Tape("1011")
    print(f"Initial tape: {tape.get_content_string()}")
    print(f"Read position 0: {tape.read(0)}")
    print(f"Read position 1: {tape.read(1)}")
    print(f"Read position 5 (uninitialized): {tape.read(5)}")
    
    tape.write(2, 0)
    print(f"After writing 0 at position 2: {tape.get_content_string()}")
    
    tape.write(10, 1)
    print(f"After writing 1 at position 10: {tape.get_content_string(-5, 15)}")


def example_simple_computation():
    """A very simple computation example"""
    print("\n" + "=" * 60)
    print("Example: Simple Computation")
    print("=" * 60)
    
    # A program that writes alternating pattern
    program = {
        ('start', 0, 0, 0): ('write_1', 0, 0, 1, 'R'),
        ('start', 1, 0, 0): ('write_1', 1, 0, 1, 'R'),
        ('write_1', 0, 0, 1): ('write_0', 0, 0, 0, 'R'),
        ('write_1', 1, 0, 1): ('write_0', 1, 0, 0, 'R'),
        ('write_0', 0, 0, 0): ('write_1', 0, 0, 1, 'R'),
        ('write_0', 1, 0, 0): ('write_1', 1, 0, 1, 'R'),
    }
    
    machine = InfiniteTimeTuringMachine(program)
    machine.set_input("101")
    
    print("Program: Writes alternating 1,0,1,0,... pattern")
    print(f"Input: {machine.input_tape.get_content_string()}")
    machine.run(max_steps=15, verbose=False)
    print(f"Output: {machine.output_tape.get_content_string()}")


def example_state_tracking():
    """Demonstrate state and snapshot tracking"""
    print("\n" + "=" * 60)
    print("Example: State Tracking")
    print("=" * 60)
    
    # Simple program that counts steps
    program = {
        ('start', 0, 0, 0): ('count', 0, 1, 0, 'R'),
        ('count', 0, 0, 0): ('count', 0, 1, 0, 'R'),
        ('count', 0, 1, 0): ('count', 0, 0, 0, 'R'),
    }
    
    machine = InfiniteTimeTuringMachine(program)
    machine.set_input("111")
    
    print("Running machine...")
    machine.run(max_steps=8, verbose=False)
    
    print(f"Final state: {machine.current_state}")
    print(f"Head position: {machine.head_position}")
    print(f"Time steps: {machine.time}")
    print(f"Number of snapshots: {len(machine.snapshots)}")
    
    if machine.snapshots:
        print(f"First snapshot time: {machine.snapshots[0].time}")
        print(f"Last snapshot time: {machine.snapshots[-1].time}")


if __name__ == "__main__":
    print("Infinite Time Turing Machine - Examples")
    print("\nNote: These are simplified examples for demonstration.")
    print("Full ITTM programs can be much more complex and run for")
    print("transfinite ordinal time.\n")
    
    example_tape_operations()
    example_simple_computation()
    example_state_tracking()
    example_binary_addition()

